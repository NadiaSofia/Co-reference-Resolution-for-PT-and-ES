# Co-reference-Resolution-for-PT-and-ES

Base Model:
  - Contains an adaptation of NeuralCoref for Portuguese and Spanish.
  - Cross-Lingual Co-Reference Resolution Model.
  - Cross-Lingual MUSE words embeddings for the Portuguese and Spanish train data, aligned in a common vector space, available in folder 'weights'.
  - Train, Test and Eval folders contain the corresponding data from Corref-PT and Ancora-CO-ES used for training, testing and development purposes.
  
  - The model is prepared to do word embeddings post-processing through PCA Projection, D=3. Arg --pca in conllparser.py default to 0, can be changed.
  - The model is also prepared to read other Portuguese and Spanish corpora, namely from Data Augmentation, as long as their names end in "-PT" ou "-ES", respectively.
  - The model is prepared to do mention detection. Arg --gold_mentions in conllparser.py default to 1, can be changed.
  - The model is prepared to run in a monolingua scenario. Args --pt and --es in conllparser.py to 0 or 1 according to the language to train, --crosslingual in conllparser.py to 0 
  
Optimizd Model:
  - Contains an adaptation of NeuralCoref for Portuguese and Spanish.
  - Cross-Lingual Co-Reference Resolution Model.
  - Cross-Lingual MUSE words embeddings for the Portuguese and Spanish train data, aligned in a common vector space, available in folder 'weights'.
  - Train, Test and Eval folders contain the corresponding data from Corref-PT and Ancora-CO-ES, and the corresponding translations for the oppositve language, as a data          augmentation proposal. Used for training, testing and development purposes.
  
  - The model is prepared to do word embeddings post-processing through PCA Projection, D=3. Arg --pca in conllparser.py default to 1, can be changed.
  - The model includes an extra single mention feature, representing [mention] and [span from 5 words before to 5 words after the mention] with Multilingal Distilled mUSE contextual embeddings.
  - The model is also prepared to read other Portuguese and Spanish corpora, namely from Data Augmentation (already included in the folders), as long as their names end in "-PT" ou "-ES", respectively.
  - The model is prepared to do mention detection. Arg --gold_mentions in conllparser.py default to 1, can be changed.
  - The model is prepared to run in a monolingua scenario. Args --pt and --es in conllparser.py to 0 or 1 according to the language to train, --crosslingual in conllparser.py to 0 
  
Running the model:
  - To prepare the data and compute the features, do:
    - python -m neuralcoref.train.conllparser --path ~/.../train/
    - python -m neuralcoref.train.conllparser --path ~/.../test/
    - python -m neuralcoref.train.conllparser --path ~/.../eval/
  - To train the model, do:
    - python -m neuralcoref.train.learn --train ~/.../train/ --eval ~/.../eval/ --test ~/.../test/  
  - To test the model with previous checkpoint file, do:
    - python -m neuralcoref.train.learn --train ~/.../train/ --eval ~/.../eval/ --test ~/.../test/ --checkpoint_file ~/.../file --test_model 1
  
  
