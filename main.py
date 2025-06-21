from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2',pad_token_id = tokenizer.pad_token_id )

input_id = 'World Animal Day Blog Hop'

token_txt = tokenizer(input_id,return_tensors ='pt',
                      padding=True, 
                      truncation=True, 
                      return_attention_mask=True)

gen_txt = model.generate(token_txt['input_ids'],attention_mask=token_txt['attention_mask'],max_length=500,num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

decode_txt = tokenizer.decode(gen_txt[0],skip_special_tokens=True)

print(decode_txt)