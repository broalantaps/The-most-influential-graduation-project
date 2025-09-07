## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import logging
import math
import os
import random
from typing import Any, List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from rich.console import Console
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.functional import gelu
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
console = Console()

class Compressor(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        max_length: int,
        embed_len: int,
        stage:int = 1,
        is_train: bool = False,
        use_lora: bool = False,
        lora_r: int = 64,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_adapter_path: str = None,
        gradient_checkpoint: bool = False,
    ):
        super(Compressor, self).__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
        )
        
        new_token_dict = {'additional_special_tokens':[f'<mem_{i}>' for i in range(64)]}
        num_added_tokens = self.tokenizer.add_special_tokens(new_token_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.mem_ids = [self.tokenizer.convert_tokens_to_ids(f'<mem_{i}>') for i in range(embed_len)]

        self.device = device
        self.stage = stage
        self.is_train = is_train
        self.max_length = max_length
        self.embed_len = embed_len
        self.model.to(self.device)
        
        if use_lora:
            # When using llama3-8B-Instruct as compressor, set the pad token to <|reserved_special_token_0|>
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "<|reserved_special_token_0|>"
                
            perf_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["embed_tokens","q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","lm_head"],
            )

            if lora_adapter_path is None:
                self.model = get_peft_model(self.model, perf_config)
                self.model.print_trainable_parameters()
                print("lora adapter weigts initialize randomly!!!")
            else:
                adapter_name = "pcc_adapter"
                self.model.load_adapter(lora_adapter_path, adapter_name=adapter_name)

            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

        else:
            # When using gpt2-large as compressor, set the pad token to unk_token
            self._set_grad_mode(is_train)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            
        # self.model.enable_input_require_grads()
        if gradient_checkpoint:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


    def _set_grad_mode(self, is_train) -> None:
        if not is_train:
            self.model.eval()
        else:
            self.model.train()
        
        for param in self.model.parameters():
            param.requires_grad = is_train
                
    def forward(
        self, 
        input_ids: torch.tensor,
    ):
        with torch.no_grad():
            mem_ids_tensor = torch.tensor(self.mem_ids).unsqueeze(0).repeat(input_ids.size(0), 1).to(self.device)
            input_ids_ = torch.cat((input_ids,mem_ids_tensor),dim=1).to(self.device)
            
            attention = torch.full((input_ids_.size(0),input_ids_.size(1)),1).to(self.device)
        with autocast(dtype=torch.bfloat16):
            text_embedding = self.model(input_ids=input_ids_,attention_mask=attention, output_hidden_states=True)
        embedding = text_embedding.hidden_states[-1][:,-self.embed_len:,:]
        
        return embedding



class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) 
        self.variance_epsilon = eps  

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)  


class Converter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        embed_len: int,
        llm_dim: int
    ):
        super(Converter, self).__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = llm_dim
        
        self.embed_len = embed_len
        self.RMSNorm = LlamaRMSNorm(embed_dim)
        
        self.dense_in = nn.Linear(embed_dim, llm_dim)
        self.dense_out = nn.Linear(llm_dim, llm_dim)
        
        self.print_trainable_parameters()
        
    def print_trainable_parameters(self):
        trainable_param = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
        print(f"Converter trainable parameters: {trainable_param}, All parameters: {all_param}")
        
    def forward(
        self, 
        embeddings: torch.Tensor
    ):
        embeddings = self.RMSNorm(embeddings)
        embeddings = embeddings.to(torch.bfloat16)  
        x = self.dense_in(embeddings)
        x = self.dense_out(gelu(x))
        x = x.to(torch.float32)
        return x

 
class Decoder(nn.Module):
    def __init__(
        self, 
        model_name_or_path: str, 
        device: str, 
        max_length: int,
        stage: int = 1,
        bf16: bool = True, 
        is_train: bool = False,
        embed_len: int = 64,
        gradient_checkpoint: bool = False,
    ):
        self.embed_len = embed_len
        super(Decoder, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        
        num_added = None
        if model_name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            self.tokenizer.eos_token_id = 128001
            self.tokenizer.pad_token_id = 128002  #<|reserved_special_token_0|>
            patch = torch.load("model/patch/llama3_8b_special_token_patch.pt")
            new_tokens = patch["tokens"]
            embed_weight = patch["embedding"].to(self.model.dtype)
            lm_head_weight = patch["lm_head"].to(self.model.dtype)

            special_tokens_dict = {"additional_special_tokens": new_tokens}
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            assert num_added == len(new_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

            new_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
            for i, token_id in enumerate(new_token_ids):
                self.model.get_input_embeddings().weight.data[token_id] = embed_weight[i]
                self.model.lm_head.weight.data[token_id] = lm_head_weight[i]

        

        self.stage = stage
        
        self.is_train = is_train

        self._set_grad_mode(is_train)
        
        self.device = device
        self.model.to(self.device)
        
        self.max_length = max_length
        
            
        self.bos_token_id = self.tokenizer.bos_token_id
        self.bos_embedding = self.model.get_input_embeddings()(torch.tensor([self.bos_token_id]).to(self.device))
        
        self.mem_token_id = self.tokenizer.convert_tokens_to_ids('<mem>')
        self.mem_embedding = self.model.get_input_embeddings()(torch.tensor([self.mem_token_id]).to(self.device))
        self.end_mem_token_id = self.tokenizer.convert_tokens_to_ids('</mem>')
        self.end_mem_embedding = self.model.get_input_embeddings()(torch.tensor([self.end_mem_token_id]).to(self.device))
        self.ae_token_id = self.tokenizer.convert_tokens_to_ids('<ae>')
        self.ae_embedding = self.model.get_input_embeddings()(torch.tensor([self.ae_token_id]).to(self.device))
        
        console.print(f'Successful add {num_added} tokens. New vocabulary size is {len(self.tokenizer)}'
                      ,style='bold yellow')
        
        if gradient_checkpoint:
            self.model.gradient_checkpointing_enable()
    
    def _set_grad_mode(self, is_train):
        if not is_train:
            self.model.eval()
        else:
            self.model.train()
        
        for param in self.model.parameters():
            param.requires_grad = is_train

    def _get_segment_mem(self, input_embedding):
        bos_embedding = self.bos_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
        mem_embedding = self.mem_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
        end_mem_embedding = self.end_mem_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)

        seg_len = math.ceil(input_embedding.size(1)/self.embed_len)
        # Adjust the shape according to your needs
        cat_embedding = torch.zeros((input_embedding.size(0),input_embedding.size(1)+seg_len*2,input_embedding.size(2))).to(self.device)
        for i in range(seg_len):
            bos_index = (i*(self.embed_len+2))
            end_index = min((i+1) * (self.embed_len+2) - 1,cat_embedding.size(1)-1)

            cat_embedding[:,bos_index,:] = mem_embedding.squeeze(1)
            cat_embedding[:,end_index,:] = end_mem_embedding.squeeze(1)
            cat_embedding[:,bos_index+1:end_index,:] = input_embedding[:,i*self.embed_len:min((i+1)*self.embed_len,input_embedding.size(1)),:]
         
        return torch.cat((bos_embedding,cat_embedding),dim=1)
  
    def generate(self,input_embedding,prompt_text,max_new_token=10):
        self.model.eval()
        with torch.no_grad(): 
            encoder_prompt_text = self.tokenizer(
                    prompt_text,
                    padding="longest",
                    add_special_tokens=False,
                    return_tensors='pt'
            ).to(self.device)
            prompt_text_ids = encoder_prompt_text['input_ids']
            # prompt_text_attention_mask = encoder_prompt_text['attention_mask']
            prompt_text_embedding = self.model.get_input_embeddings()(prompt_text_ids).to(self.device)

            bos_embedding = self.bos_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
            mem_embedding = self.mem_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
            end_mem_embedding = self.end_mem_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
            seg_len = math.ceil(input_embedding.size(1)/self.embed_len)
            
            cat_embedding = torch.zeros((input_embedding.size(0),input_embedding.size(1)+seg_len*2,input_embedding.size(2))).to(self.device)
            for i in range(seg_len):
                bos_index = (i*(self.embed_len+2))
                end_index = min((i+1) * (self.embed_len+2) - 1,cat_embedding.size(1)-1)
                cat_embedding[:,bos_index,:] = mem_embedding.squeeze(1)
                cat_embedding[:,end_index,:] = end_mem_embedding.squeeze(1)
                cat_embedding[:,bos_index+1:end_index,:] = input_embedding[:,i*self.embed_len:min((i+1)*self.embed_len,input_embedding.size(1)),:]

            input_embedding = torch.cat((bos_embedding,cat_embedding),dim=1)

            embedding = torch.cat((input_embedding,prompt_text_embedding),dim=1).to(self.device)
            
            output = embedding.clone()
            past_key_values = None
            generate_text = []
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            with autocast():
                for i in range(max_new_token):
                    out = self.model(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                    logits = out.logits[:, -1, :len(self.tokenizer)]
                    past_key_values = out.past_key_values

                    next_token_id = torch.argmax(logits,dim=-1)
                    output = self.model.get_input_embeddings()(next_token_id.unsqueeze(1).to(self.device))
                    generate_text.append(next_token_id.item())
                    if next_token_id.item() in terminators:
                        break

            output_text = self.tokenizer.decode(generate_text,skip_special_tokens=True)
            return output_text
            

    def forward(
        self,
        input_embedding: torch.tensor,
        prompt_text: Union[str, List[str]]=None,
        llm_ids: Union[int,List[int]]=None,
        labels_ids: Union[int,List[int]]=None,
        next_ids: Union[int,List[int]]=None,
        task_type: Union[str, List[str]]=None
    ):
        if task_type not in ["ae","next_token","rag"]:
            raise ValueError("task_type must be 'ae' or 'next_token' or 'rag', but got {task_type}")
        
        with torch.no_grad():
            target_text_ids = torch.tensor(llm_ids).to(self.device) if task_type in ["ae","rag"]  else torch.tensor(next_ids).to(self.device)
            if target_text_ids.dim() == 1:
                target_text_ids = target_text_ids.unsqueeze(0).to(self.device)

            target_text_attention_mask = (target_text_ids != self.tokenizer.pad_token_id).long().to(self.device)
            target_text_embedding = self.model.get_input_embeddings()(target_text_ids).to(self.device)
            
            if task_type == "ae":
                encoder_prompt_text = self.tokenizer(
                        prompt_text,
                        padding="longest",
                        add_special_tokens=False,
                        return_tensors='pt'
                ).to(self.device)
                prompt_text_ids = encoder_prompt_text['input_ids']
                prompt_text_attention_mask = encoder_prompt_text['attention_mask']
                prompt_text_embedding = self.model.get_input_embeddings()(prompt_text_ids).to(self.device)
                
        with autocast(dtype=torch.bfloat16):
            bos_embedding = self.bos_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
            mem_embedding = self.mem_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
            end_mem_embedding = self.end_mem_embedding.unsqueeze(0).repeat(input_embedding.size(0),1,1)
            seg_len = math.ceil(input_embedding.size(1)/self.embed_len)
            # Adjust the shape according to your needs
            cat_embedding = torch.zeros((input_embedding.size(0),input_embedding.size(1)+seg_len*2,input_embedding.size(2))).to(self.device)
            for i in range(seg_len):
                bos_index = (i*(self.embed_len+2))
                end_index = min((i+1) * (self.embed_len+2) - 1,cat_embedding.size(1)-1)
                cat_embedding[:,bos_index,:] = mem_embedding.squeeze(1)
                cat_embedding[:,end_index,:] = end_mem_embedding.squeeze(1)
                cat_embedding[:,bos_index+1:end_index,:] = input_embedding[:,i*self.embed_len:min((i+1)*self.embed_len,input_embedding.size(1)),:]

            # cat llm's bos token
            input_embedding = torch.cat((bos_embedding,cat_embedding),dim=1)
           
        embedding_attention_mask = torch.ones((input_embedding.size(0),input_embedding.size(1))
                                              ).to(self.device)
            
        if task_type == "ae":
            attention_mask = torch.cat((embedding_attention_mask,prompt_text_attention_mask,target_text_attention_mask), 
                                            dim=1).to(self.device)
            embedding = torch.cat((input_embedding,prompt_text_embedding,target_text_embedding),
                                                dim=1).to(self.device)
        elif task_type == "rag":
            attention_mask = torch.cat((embedding_attention_mask,target_text_attention_mask), 
                                            dim=1).to(self.device)

            embedding = torch.cat((input_embedding,target_text_embedding),
                                                dim=1).to(self.device)
            
            
        else:
            attention_mask = torch.cat((embedding_attention_mask,target_text_attention_mask), 
                                            dim=1).to(self.device)
            embedding = torch.cat((input_embedding,target_text_embedding),
                                                dim=1).to(self.device)
        
        targets = target_text_ids.masked_fill(target_text_ids == self.tokenizer.pad_token_id,-100)
        
        embed_target =  torch.ones(embedding_attention_mask.size(), dtype=torch.long).to(self.device)
        
        if task_type == "ae":
            prompt_target = torch.ones(prompt_text_attention_mask.size(),dtype=torch.long).to(self.device)
            empty_target = torch.cat((embed_target,prompt_target),dim=1).to(self.device)
        else:
            empty_target = embed_target
        
        empty_target =(
           empty_target.fill_(-100)
        )
        if task_type == "rag":
            labels_ids_tensor = torch.tensor(labels_ids).to(self.device)

        targets_ = torch.cat((empty_target,targets),dim=1).to(self.device) if task_type in ["ae","next_token"] else torch.cat((empty_target,labels_ids_tensor),dim=1).to(self.device)
        
        with autocast(dtype=torch.bfloat16):
            output = self.model(
                inputs_embeds = embedding,
                attention_mask = attention_mask,
                return_dict=True,
                labels = targets_,
            )
            
        sum_loss = output.loss.mean()

        return {"loss":sum_loss, "logits":output.logits, "target":targets_, "ppl_loss":0}



class PCC(nn.Module):
    def __init__(self, args):
        super(PCC, self).__init__()
        self.segment_length = args.segment_length
        self._device = args.device
        self.args = args
        
        assert args.stage in [1,2], "stage must be 1 or 2"
        use_lora = getattr(args, 'use_lora', False)
        if not use_lora:
            self.compressor = Compressor(
                model_name_or_path=args.compress_model,
                device=args.device,
                embed_len=args.embed_len,
                max_length=512,
                is_train=True,
                gradient_checkpoint=args.compressor_gradient_checkpoint
            )
        else:
            self.compressor = Compressor(
                model_name_or_path=args.compress_model,
                device=args.device,
                embed_len=args.embed_len,
                max_length=512,
                is_train=True,
                use_lora=args.use_lora,
                lora_adapter_path=args.adapter_model,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                gradient_checkpoint=args.compressor_gradient_checkpoint
            )
        
        self.decoder = Decoder(
            model_name_or_path=args.decoder_model,
            stage=args.stage,
            device=args.device,
            max_length=2048,
            is_train=False,
            embed_len=args.embed_len,
            gradient_checkpoint=args.decoder_gradient_checkpoint
        )
    
        self.converter = Converter(
            embed_dim=self.compressor.model.config.hidden_size,
            embed_len=args.embed_len,
            llm_dim=self.decoder.model.config.hidden_size
        )
        
        
        if args.converter_model is not None:
            if os.path.exists(args.converter_model):    
                self.converter.load_state_dict(torch.load(args.converter_model))
                print(f"Load converter successfully from {args.converter_model}")
            else:
                converter_model_path = hf_hub_download(
                    repo_id=args.converter_model,
                    filename='memory_converter.bin'
                )
                print(f"converter.bin saved to {converter_model_path}")
                self.converter.load_state_dict(torch.load(converter_model_path))
                print(f"Load converter successfully from {args.converter_model}")
        else:
            console.print("No converter model loaded, the param of converter will be initialized randomly.", style="bold red")
    
    def generate(
        self, 
        compress_ids:Union[int,List[int]],
        prompt_text: Union[str, List[str]],
        max_new_token: int,
    ):
        # set model's mode to eval
        self.decoder.model.eval()
        self.compressor.model.eval()
        self.converter.eval()
        input_ids = torch.tensor(compress_ids).unsqueeze(0).to(self._device)
        
        bsz, input_len = input_ids.size(0), input_ids.size(1)
        
        segment = self.segment_length 
        
        # get sum segments
        segment_len = math.ceil(input_len / segment)
        
        # be used to store concatenated embedding
        text_embedding = None
        for i in range(segment_len):
            with autocast(dtype=torch.bfloat16):
                # segment interval
                segment_begin = i * segment
                segment_end = ((i+1)*segment) if ((i+1)*segment) < input_len else input_len
                
                segment_ids = input_ids[:, segment_begin:segment_end]
                
                # get per segment's memory
                memory = self.compressor(segment_ids)
                text_embedding: Any | torch.Tensor = memory if text_embedding is None else torch.cat(
                                                    (text_embedding,memory
                                                ),dim=1)
                

        # memory_embed's shape equal to [bsz,embed_len*num_segment,llm_dim]
        memory_embed = self.converter(text_embedding)
        del text_embedding
        generate_text = self.decoder.generate(memory_embed, prompt_text, max_new_token)
        return generate_text
    
    def forward(
        self, 
        compress_ids:Union[int,List[int]],
        llm_ids:Union[int,List[int]],
        labels_ids:Union[int,List[int]]=None,
        prompt_text: Union[str, List[str]]=None, 
        next_ids:Union[int,List[int]]=None,
        task_type=None,
        get_embedding=False
    ):
        input_ids = torch.tensor(compress_ids).to(self._device)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0).to(self._device)

        bsz, input_len = input_ids.size(0), input_ids.size(1)
        
        segment = self.segment_length 
        
        segment_len = math.ceil(input_len / segment)
        
        text_embedding = None
        
        for i in range(segment_len):
            with autocast(dtype=torch.bfloat16):
                segment_begin = i * segment
                segment_end = ((i+1)*segment) if ((i+1)*segment) < input_len else input_len
                
                segment_ids = input_ids[:, segment_begin:segment_end]
                
                model_instance = self.compressor(segment_ids)
                text_embedding = model_instance if text_embedding is None else torch.cat(
                                                    (text_embedding,model_instance
                                                ),dim=1)
                

        embed = self.converter(text_embedding)
        if get_embedding:
            return embed
        if self.args.stage == 1:
            if task_type is None:
                thresold = random.random()
                if thresold > self.args.next_token_ratio:
                    task_type = "ae"
                else:
                    task_type = "next_token"
        elif self.args.stage == 2:
            task_type = "rag"
        else:
            raise ValueError("stage must be 1 or 2")

        loss_dict = self.decoder(input_embedding=embed,prompt_text=prompt_text,llm_ids=llm_ids,labels_ids=labels_ids,
                             next_ids=next_ids,task_type=task_type)
        return loss_dict 