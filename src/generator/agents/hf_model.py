from src.agent import Agent
from transformers import GenerationConfig,\
    AutoConfig,\
    AutoModelForSeq2SeqLM,\
    AutoModelForCausalLM,\
    AutoTokenizer
import torch
from tqdm import tqdm


load_in_8bit_models = ["llama2-70b-chat","llama2-70b"]

llama_ckpt = {
    "llama2-7b-chat":"/fs/ess/PAA0201/llama2/llama-2-7b-chat",
    "llama2-7b":"/fs/ess/PAA0201/llama2/llama-2-7b",
    "llama2-13b-chat":"/fs/ess/PAA0201/llama2/llama-2-13b-chat",
    "llama2-13b":"/fs/ess/PAA0201/llama2/llama-2-13b",
    "llama2-70b-chat":"/fs/ess/PAA0201/llama2/llama-2-70b-chat",
    "llama2-70b":"/fs/ess/PAA0201/llama2/llama-2-70b"
}
llama_spm = "/fs/ess/PAA0201/llama2/tokenizer.model"


class HFAgent(Agent):
    def __init__(self,model_name, batch_size = 1,**kwargs) -> None:
        
        super().__init__(**kwargs)
        
        # if llama, would try to load llama ckpt first and then try hf one
        if "llama" in self.name:
            self.is_llama = True 
        else:
            self.is_llama = False

        self.logger.info(f"self.is_llama: {self.is_llama}")


        self.logger.warning(f"prompt name is {self.prompt_name}, and you need to double check this is what you want as"\
                         "diff models may need diff format")
        self.model_name = model_name

        if self.is_llama:
            self.gen_args_llama = {"temperature":1,"top_p":1,"max_gen_len":64}
            for key in kwargs:
                if self.gen_args_llama.get(key,None) is not None:
                    self.gen_args_llama[key] = kwargs[key]

        _t = GenerationConfig()
        gen_args = {}
        for key in kwargs:
            if hasattr(_t,key):
                self.logger.info(key)
                gen_args[key] = kwargs[key]
        print(gen_args)
        self.gen_args = GenerationConfig(**gen_args)

        self.batch_size = batch_size
        self.load_model()
        
    def load_model(self):

        use_hf = True
        if self.is_llama:

            try:
                from llama import Llama, Dialog
                print("here")
                self.generator = Llama.build(
                    ckpt_dir=llama_ckpt[self.name],
                    tokenizer_path=llama_spm,
                    max_seq_len=512,
                    max_batch_size=self.batch_size,
                )
                print("here2")
                use_hf = False

            except:
                use_hf = True

            if use_hf:
                self.logger.info("Load llama from hf")
            else:
                self.logger.info("Load llama from pt")

        if use_hf:

            try:
                config = AutoConfig.from_pretrained(self.model_name, token= True)
            except:
                raise Exception()
            
            load_in_8bit = True if self.name in load_in_8bit_models else False

            architectures = config.architectures[0]
            if "Conditional" in architectures:
                self.only_decoder = False
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map = "auto", trust_remote_code = True, token= True,load_in_8bit = load_in_8bit)
            elif "Causal" in architectures:
                self.only_decoder = True
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto", trust_remote_code = True, token= True,load_in_8bit = load_in_8bit)
            assert self.model.can_generate, "model could not generate"
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True, token= True)
            if tokenizer.pad_token is None or "Causal" in architectures:
                self.logger.info("Use left padding")
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"
            self.tokenizer = tokenizer
            self.model.eval()

        self.use_hf = use_hf



    @torch.no_grad()
    def inference(self, historys) -> str:
        if self.use_hf:
            pbar = tqdm(total=len(historys),desc="Inference hf")
            output_batch_final = []
            for batch in Agent.batch_sample(historys,self.batch_size):
                self.logger.info("batch")
                tokenized_batch = self.tokenizer(batch, padding= True,return_tensors="pt").to(self.model.device)
                self.logger.info("tokenized_batch")
                self.logger.info(self.gen_args)
                output_batch = self.model.generate(**tokenized_batch,generation_config = self.gen_args)
                self.logger.info("output_batch")
                if self.only_decoder:
                    output_batch_cut = output_batch[:,tokenized_batch["input_ids"].shape[-1]:]
                else:
                    output_batch_cut = output_batch
                output_batch_final.extend(self.tokenizer.batch_decode(output_batch_cut,skip_special_tokens=True))
                pbar.update(len(batch))
            return output_batch_final
        
        if not self.use_hf and self.is_llama:


            pbar = tqdm(total=len(historys),desc="Inference llama")
            output_batch_final = []

            
            for batch in Agent.batch_sample(historys,self.batch_size):
                if self.prompt_name == "plain":
                    results = self.generator.text_completion(
                        batch,
                        **self.gen_args_llama
                    )
                    for result in results:
                        output_batch_final.append(result["generation"])
                if self.prompt_name == "llama-2":
                    results = self.generator.chat_completion(
                        batch,  # type: ignore
                        **self.gen_args_llama
                        )
                    for result in results:
                        output_batch_final.append(result["generation"]["content"]) 

                pbar.update(len(batch))
            return output_batch_final
            



        


        



