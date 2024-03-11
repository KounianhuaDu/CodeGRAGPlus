from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
    get_peft_model_state_dict
)
import torch
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from tqdm import tqdm
import json
import argparse
from datasets import load_dataset


def get_dataset(data_files, tokenizer):
    dataset = load_dataset('json', data_files)
    print('Dataset loaded')
    prompt = (
        '''
You are a powerful programmer, please continue to complete the {} function according to the requirements and function declarations. You are not allowed to modify the given code and do the completion only.\n

### Input:
The problem:
\n
{}

### Instruction:
The syntax graph of a similar code might be:\n
{}
You can refer to the above knowledge to do the completion. 

### Response:
'''.strip()
    )

    def apply_prompt_template(knowledge, question, language, answer: str):
        return {
            "prompt": prompt.format(language, knowledge, question.strip()),
            "output": answer
        }

    dataset = dataset.map(apply_prompt_template,
                          remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        len_user_prompt_tokens = (len(tokenizer(sample['prompt'],
                                                truncation=True,
                                                max_length=2048
                                                )["input_ids"]))
        
        full_tokens = tokenizer(
            sample["prompt"] + sample["output"], max_length=2048, truncation=True)["input_ids"]

        return {
            "input_ids": full_tokens,
            "attention_mask": [1] * (len(full_tokens)),
            "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:]
        }
    dataset = dataset.map(tokenize_add_label,
                          remove_columns=list(dataset.features))

    return dataset


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='logs')
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--model_path", type=str, default='./models/CodeLlama')
parser.add_argument('--model', type=str, default='CodeLlama')
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument('--train_size', type=int, default=65536)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--load_in_8bit", action='store_true',
                    help="Load model 8 bit.")

args = parser.parse_args()
args.data_path = './data'
args.output_path = './trained_models'


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

# Dataset
train_dataset = get_dataset(args.data_path, tokenizer,)

# Model
model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                             load_in_8bit=args.load_in_8bit,
                                             device_map='auto',
                                             torch_dtype=torch.float16)
model.train()

MICRO_BATCH_SIZE = args.per_device_eval_batch_size
BATCH_SIZE = min(args.total_batch_size, args.train_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
LEARNING_RATE = args.lr
CUTOFF_LEN = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.val_size  # 2000
OUTPUT_DIR = args.output_path

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"]
)

if args.load_in_8bit:
    model = prepare_model_for_int8_training(model)

model = get_peft_model(model, peft_config)


config = {
    'lora_config': peft_config,
    'learning_rate': LEARNING_RATE,
    'num_train_epochs': 1,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'per_device_train_batch_size': MICRO_BATCH_SIZE,
    'gradient_checkpointing': False
}

# Define training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    bf16=True,
    logging_strategy="steps",
    logging_steps=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    max_steps= = MAX_STEPS,
    logging_dir=args.log,
    optim="adamw_torch",
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

# Define trainer
trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            callbacks=EarlyStoppingCallback(2)
        )
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)
    
print("Start training...")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)