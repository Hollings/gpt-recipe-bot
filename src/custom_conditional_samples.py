#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import sys
import random
import model, sample, encoder
import wikipedia

# print(sys.argv[1])
# sys.exit(0);
def random_food():
    line = (random.choice(list(open('foods.txt'))))
    return line.title()

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=None,
    length=1000,
    temperature=0.8,
    top_k=40,
):
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
            food = random_food();
            raw_text = food + " Recipe:"
            try:
                wiki = wikipedia.page(food)
                image = "<img style='max-height:300px;src='" + wiki.images[0] + "'>"
            except:
                image = ""
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    text = text.split("<|endoftext|>")[0]
                    text = text.replace("\n","<br>")
                    text_file = open("/var/www/recipe/index.html", "w")
                    text = "<div style='width:66%;position:absolute;left:16%'><h1>" + raw_text + "</h1>" + image + "<br>" + str(text) + "</div>"
                    text_file.write(text)
                    text_file.close()
                    # print(str(text))

if __name__ == '__main__':
    fire.Fire(interact_model)

