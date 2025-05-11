# Exploring Show, Attend, Tell Attention Mechanisms for Image Captioning

## Introduction
****
Image captioning explores the space defined by the intersection of Natural Language Processing (NLP) and Computer Vision (CV). Generating a caption given an image, especially in zero-shot cases, is a difficult task due to the need for semantic alignment between two very different spaces and the inherent subjectivity in what is of importance in an image.


The present paper directly iterates on the approach given by Vinyal's et. al 2015 by introducing an attention mechanism to this CNN-RNN paradigm, hence the name **Show Attend Tell**. In addition to the traditional $h_{t-1}$, $y_t$ inputs to the LSTM gates, a context vector $z_t$ is computed using one of the proposed attention variants and is concatenated as an additional input at timestep $t$. In turn, the decoder RNN is able to learn where it should "look" in order to generate the $t$'th caption word. This dynamic attention mechanism helps mitigate the bottlenecking problem of RNN architectures, achieving state-of-the-art BLEU and METEOR scores in 2016, while also providing for increased interpretability.

## Chosen Result
TODO

## GitHub Contents 

Explanations of each of the different files used in the project:

- `run_sat.ipynb` - Python notebook which user can interact with to train, test, and visualize the results of our model. It interacts with both the code and data directories.

### Code Directory
- `datasets.py` - Vocabulary, Flickr8kDataset, collate_fn, LengthBasedBatchSampler, helpers
- `models.py` - Neural architectures: EncoderCNN, Attention, DecoderRNN, ShowAttendTell
- `training.py` - Training / evaluation routines, checkpoints, metric helpers
- `visualize.py` - Utilities for finding samples and plotting attention maps

### Data Directory
- `kaggle_cache/versions/2/Flickr8kDataset` - Contains the Flickr8k image dataset used for training and testing
- `nltk_data/` - Contains NLTK resources including:
  - `corpora/` - Natural language corpora for text processing
  - `tokenizers/` - Text tokenization resources

## Re-implementation Details
TODO

## Reproduction Steps

1. Clone this repository.

2. Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
3. Open the notebook. It is organized into clearly labeled sections.

4. Run each section sequentially to reproduce the results.


## Running the Notebook

The notebook `run_sat.ipynb` is organized into 6 main sections:

1. **Setup (00-setup)**: Initializes the project environment, sets up paths, and configures NLTK data directories
2. **Imports (01-imports)**: Imports necessary modules and sets up the device (CPU/GPU)
3. **Data (02-data)**: Downloads and prepares the Flickr8k dataset, sets up data transformations
4. **Model (03-model)**: Initializes the ShowAttendTell model with specified parameters
5. **Train (04-train)**: Handles the model training and evaluation process
6. **Visualize (05-visualize)**: Generates sample captions and creates attention visualization plots

Each section is clearly labeled with comments and can be run sequentially to reproduce the results.

## Example Outputs

After completing training, the visualization section will output where our model is attending and its caption attempts. An example is included below:

[TODO ADD EXAMPLE VISUALIZATION HERE]

## Results/Insights
TODO

## Conclusion
TODO

## References

1. Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., Bengio, Y. (2015). "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." arXiv preprint arXiv:1502.03044. M. Hodosh, P. Young and J. Hockenmaier (2013) 
2. "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899 http://www.jair.org/papers/paper3994.html
3. Hannah Li, Sivaraman K S. "Show, Attend, and Tell Neural Image Caption Generation with Visual Attention". [Slides](https://www.cs.rice.edu/~vo9/vislang/2017/slides/sivahaina.pdf)
4. Halfling Wizard. (2021). "Show, Attend And Tell - Paper Explained." YouTube. https://www.youtube.com/watch?v=y1S3Ri7myMg 

## Acknowledgements

We would like to thank the authors of the original papers for their groundbreaking work in attention-based image captioning. Special thanks to [3Blue1Brown's video on attention mechanisms](https://www.youtube.com/watch?v=y1S3Ri7myMg) for providing an excellent intuitive explanation of the concept.

We would like to wish our moms a happy Mother's Day!