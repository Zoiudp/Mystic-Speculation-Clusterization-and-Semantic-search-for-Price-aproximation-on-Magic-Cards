
import torch
from transformers import DistilBertForSequenceClassification


def model_prep(dfs):
    data_texts = dfs['text'].astype(str).to_list()
    data_labels = dfs['classified_text'].to_list()
    train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size = 0.8, random_state = 0 )

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation = True, padding = True  )

    val_encodings = tokenizer(val_texts, truncation = True, padding = True )

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))


    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))

    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

    from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments


    training_args = TFTrainingArguments(
        output_dir='/results',
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=50000,
        weight_decay=1e-5,
        logging_dir='/logs',
        eval_steps=1000
    )

    with training_args.strategy.scope():
        trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 20 )


    trainer = TFTrainer(
        model=trainer_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.evaluate()
    save_directory = "/saved_models"

    model.save_pretrained(save_directory)

    tokenizer.save_pretrained(save_directory)

    tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)

    model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

    test_text = "Flying, vigilance At the beginning of your upkeep, investigate once for each opponent who has more cards in hand than you. "

    #test_text

    predict_input = tokenizer_fine_tuned.encode(
    test_text,
    truncation = True,
    padding = True,
    return_tensors = 'tf'
    )

    output = model_fine_tuned(predict_input)[0]

    prediction_value = tf.argmax(output, axis = 1).numpy()[0]

    print(prediction_value)


    tokenizer_fine_tuned_pt = DistilBertTokenizer.from_pretrained(save_directory)


    model_fine_tuned_pt = DistilBertForSequenceClassification.from_pretrained(save_directory, from_tf = True )

    predict_input_pt = tokenizer_fine_tuned_pt.encode(test_text, truncation = True, padding = True, return_tensors = 'pt' )

    ouput_pt = model_fine_tuned_pt(predict_input_pt)

    prediction_value_pt = torch.argmax(ouput_pt[0], dim = 1 ).item()

    print(prediction_value_pt)

    

