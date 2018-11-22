# A kernel of Mercari Price Suggestion Challenge implemented with Keras
<pre>
            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    Version 2, December 2004

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE FUCK YOU WANT TO.
</pre>

### Requirements
* Python >= 3.6
* TensorFlow >= 1.6
* Keras >= 2.0.0

### Model Plot
![Model Plot](https://github.com/kaitolucifer/mercari-price-suggestion-challenge-keras/blob/master/model.png)
You can download the dataset from [here](https://www.kaggle.com/c/mercari-price-suggestion-challenge)</br>
I put all categorical features into different embedding layers.</br>
And for the text featrues, I change them to sequence data and put them into embedding layers which initialized by [glove.6B(Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download))](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip).</br>
Then use TextCNN to extract their features.(Also tried RNN)</br>
Sort of joint learning.</br>
