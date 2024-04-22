import torch
import datasets
import pandas as pd
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments


base_model = "meta-llama/Llama-2-7b-chat-hf"

df = pd.read_csv("train_qa_dataset.csv")

df1 = df[['Question', 'Context', 'Answer 1', 'Score 1', 'Feedback 1']].rename(columns={
    'Question': 'question', 'Context': 'context', 'Answer 1': 'answer', 'Score 1': 'score', 'Feedback 1': 'feedback'})

df2 = df[['Question', 'Context', 'Answer 2', 'Score 2', 'Feedback 2']].rename(columns={
    'Question': 'question', 'Context': 'context', 'Answer 2': 'answer', 'Score 2': 'score', 'Feedback 2': 'feedback'})

result_df = pd.concat([df1, df2], ignore_index=True)


def generate_prompt(sample):
    return f"""
    ### Instruction:
    You are a Biology expert. Your task is to help students by evaluating their answers and providing feedback. Using the Context evaluate the Answer to the Question and Provide a Score and Feedback.
    
    ### Context:
    {sample["context"]}
    
    ### Question:
    {sample["question"]}
    
    ### Answer:
    {sample["answer"]}
    
    ### Score:
    {sample["score"]}
    
    ### Feedback:
    {sample["feedback"]}
    """


result_df["prompt"] = result_df.apply(generate_prompt, axis=1)

df = result_df["prompt"].copy()
df = pd.DataFrame(df)

dataset = datasets.Dataset.from_pandas(df)

del df1, df2, result_df


# Load the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
   base_model,
    quantization_config=bnb_config,
    device_map={"": 0}
)

model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


# Prepare the model for training
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)


# Training arguments
training_arguments = TrainingArguments(
    output_dir= "./output",
    overwrite_output_dir= True,
    num_train_epochs= 3,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 500,
    logging_steps= 20,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

# Train the model
trainer.train()

# Save the model
trainer.model.save_pretrained("./final_model")
