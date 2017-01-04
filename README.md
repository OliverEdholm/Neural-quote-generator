# Neural-quote-generator
A lstm-char-rnn generator generating quotes made in TFLearn.

### Examples
Here are some examples of quotes generated after training in 12 hours on a GTX 1070 and picking the checkpoint with the lowest loss, 1.2.

```
the man who in the world is so good to do the powers of men.

while we find we make famous for government.

stay was pretty played up well and beauty.

the love is the most happy hands - you do.

i prove the necessity in it.

there have been power, like production. i come about it.

i think there's a kind of love of the grand work.

some injured of work will retreat all my coping.

the front and old law is an off.
```

As you can see are the quotes not giving any wisdom but are rather very confusing and quite entertaining according to me.

This neural network can probably do much better with some mixturing of parameters and changing the layout of the network but I got what I wanted from the network so I won't go any further.

### Requirements
* Python 3.*
* TFlearn

### Usage
Run this to train the lstm-char-rnn on quotes downloaded from https://github.com/alvations/Quotables.
```
python3 quote_lstm.py
```

All checkpoints will be stored in a folder called checkpoints. When training a new model, make sure to make a backup of the charidx.pkl and quotes.txt files and the checkpoints folder.

To evaluate a checkpoint you can run.
```
python3 evaluate_lstm <checkpoints/checkpointname>
```

### Credits
Dataset taken from https://github.com/alvations/Quotables

### Other
Made by Oliver Edholm, 14 years old.
