from lib.prepareDatasetDict import createDatasetDict
from lib.show_random_elements import show_random_elements
from lib.remove_special_characters import remove_special_characters
from lib.extract_all_chars import extract_all_chars
from lib.DataCollatorCTCWithPadding import DataCollatorCTCWithPadding

import json
import numpy as np
import random
from datasets import load_metric
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

pretrained_model = "facebook/wav2vec2-base"
dataset = createDatasetDict(train_manifest_path='./manifests/train-manifest.json', val_manifest_path='./manifests/val-manifest.json', test_manifest_path='./manifests/test-manifest.json')
repo_name = f"exp/{pretrained_model}"
print(dataset)


show_random_elements(dataset["train"].remove_columns(["audio", "file"]), num_examples=10)
dataset = dataset.map(remove_special_characters)
print('[dataset after removing special characters]')
show_random_elements(dataset["train"].remove_columns(["audio", "file"]))




vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print(vocab_dict)
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)




######## saving vocabulary file ############
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)




tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


rand_int = random.randint(0, len(dataset["train"]))
print("Target text:", dataset["train"][rand_int]["text"])
print("Input array shape:", np.asarray(dataset["train"][rand_int]["audio"]["array"]).shape)
print("Sampling rate:", dataset["train"][rand_int]["audio"]["sampling_rate"])


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



model = Wav2Vec2ForCTC.from_pretrained(
    pretrained_model,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_encoder()




training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=200,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()