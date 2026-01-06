# Radio Frequency Interference Classification Project

Develop models to classify 9 types of RF interference that occurred along a section of a German highway.

## Dataset

### Example Samples
![Example RF interference samples](assets/classes_2sets.png)

### Samples Distribution
![Train/test distribution](assets/highway_ds_train_test.png)

### Source
https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2




## CNN
CNN0to8 (with and without extra data, various depths, resized, and not resized)

Neural network models produce results very similar to the CNN models shown above. They struggle with class 1,3,8 due to these classes having fewer samples and being very similar to class 0 or 2. They are however consistent with class 0,1,4,5,6,7. The idea behind the voting model is to pair several NN models with several of another 

dropuout (time/freq bins) rectange, shallow vs deep, resize
table

Use normed spectogram data

## Decision Tree CNN

boosted 
## Voting 
The idea behind the voting model is to pair a set of neural network models, that consistently struggle with classes 1,3,8 (similar to CNN) with a model that is better with 1,3,8 but struggles with classes 0,2. 

To achieve this, a set of neural network and "linear" models were trained on variations of the highway dataset. Each model, when given a sample, outputs a set of probabilities for each class. 

Two different voting techniques were utilized: unfited voting and fited voting. 
Unfited voting sums the probabilites across models at each class, creating a final set of scores for each class. The class corresponding to the largest score is chosen as the final class. Alternativly, fited voting uses the set of probabilites for each model as features for a NN that predicts the final class. 

### Superfeatures
Each sample in the highway dataset is trainsformed into 2112 features.
- Shift spectrogram values from **[-136, 0]** to **[0, 136]**
- Resize spectrogram from **512 × 243** to **128 × 64**
- Extract statistical features across rows and columns:
  - Mean
  - Standard deviation
  - Median
  - Minimum and maximum
  - Minimum and maximum locations
  - 75th and 25th percentiles
  - 90th and 10th percentiles
- Normalize all features to the range **[0, 1]**

### Example NN Model
This represents the typical results when evaluating neural network models. This has an accuracy of **77.1%** after one guess and **94.9%** after two guesses. The first guess involves picking the class coresponding to the highest probability and comparing it to the expected class. The second guess is the second highest probability.
<p align="center"> <img src="assets\nn_conf_matrix_p1.png" width="45%"> <img src="assets\nn_conf_matrix_p2.png" width="45%"></p>
These models perform similarly to the CNN-based models, despite using superfeatures rather than normalized spectrogram data, suggesting that the superfeatures adequately describe the dataset.



### Example "Linear" Model 
> A "linear" model is trained by finding the median and standard deviation at each feature, independently computed for each class. <br>
> **classes** c1 ... c9 <br>
> **features** x1 ... xn <br>
> **medians** m1 ... mn for all c <br>
> **standard deviations** s1 ... sn for all c <br>
> Given the features of a sample to evaluate, compute the z-score at each feature, independently compute for each class. <br>
> **sample z-scores** z1 ... zn for all c <br>
> Average z-scores across each class to get one score for each class. <br>
> **sample scores** b1 ... b9 <br>
> Convert scores into probabilities with: exp(-bi) / sum(exp(-b)) <br>
> **sample probabilities** p1 ... p9 <br>

This represents the typical results when evaluating "linear" models. This has an accuracy of **63.7%** after one guess and **79.8%** after two guesses. 
<p align="center"> <img src="assets\linear_conf_matrix_p1.png" width="45%"> <img src="assets\linear_conf_matrix_p2.png" width="45.6%"></p>
This model has a higher success rate with classifiying minority classes 3,4,5,6,7,8 with the tradoff of having more misclassification of 0,1,2.

### Trained Models
<table align="center" style="border-collapse: collapse; width: 100%; max-width: 100%;">
  <thead>
    <tr>
      <th style="text-align:left; border-bottom: 1px solid #ccc; padding: 8px;">Model</th>
      <th style="text-align:left; border-bottom: 1px solid #ccc; padding: 8px;">Description</th>
      <th style="text-align:center; border-bottom: 1px solid #ccc; padding: 8px;">Accuracy Guess 1</th>
      <th style="text-align:center; border-bottom: 1px solid #ccc; padding: 8px;">Accuracy Guess 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px;">NN1</td>
      <td style="padding: 8px;">NN, superfeatures, 8 epochs, 1.2 M params</td>
      <td style="text-align:center; padding: 8px;">78.2%</td>
      <td style="text-align:center; padding: 8px;">95.4%</td>
    </tr>
    <tr>
      <td style="padding: 8px;">NN2</td>
      <td style="padding: 8px;">NN, superfeatures, 8 epochs, 2.0 M params</td>
      <td style="text-align:center; padding: 8px;">78.8%</td>
      <td style="text-align:center; padding: 8px;">95.7%</td>
    </tr>
    <tr>
      <td style="padding: 8px;">NN3</td>
      <td style="padding: 8px;">NN, superfeatures, 6 epochs, 5.5 M params</td>
      <td style="text-align:center; padding: 8px;">78.0%</td>
      <td style="text-align:center; padding: 8px;">95.8%</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Linear1</td>
      <td style="padding: 8px;">Linear, superfeatures</td>
      <td style="text-align:center; padding: 8px;">63.6%</td>
      <td style="text-align:center; padding: 8px;">80.3%</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Linear2</td>
      <td style="padding: 8px;">Linear, superfeatures (no mean, std)</td>
      <td style="text-align:center; padding: 8px;">63.7%</td>
      <td style="text-align:center; padding: 8px;">79.8%</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Linear3</td>
      <td style="padding: 8px;">Linear, superfeatures (no mean, std, 90th &amp; 10th percentiles)</td>
      <td style="text-align:center; padding: 8px;">63.6%</td>
      <td style="text-align:center; padding: 8px;">80.0%</td>
    </tr>
  </tbody>
</table>

### Unfited Voter Results
<table align="center" style="border-collapse: collapse; width: 100%; max-width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center; padding: 8px;">Model</th>
      <th style="text-align:center; padding: 8px;">Accuracy Guess 1</th>
      <th style="text-align:center; padding: 8px;">Accuracy Guess 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center; padding: 8px;">Unfited Voter</td>
      <td style="text-align:center; padding: 8px;">77.7%</td>
      <td style="text-align:center; padding: 8px;">94.3%</td>
    </tr>
  </tbody>
</table>

<p align="center"> <img src="assets\unfited_voter_conf_matrix_p1.png" width="45%"> <img src="assets\unfited_voter_conf_matrix_p2.png" width="45%"></p>

### Fited Voter Results

<table align="center" style="border-collapse: collapse; width: 100%; max-width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center; padding: 8px;">Model</th>
      <th style="text-align:center; padding: 8px;">Accuracy Guess 1</th>
      <th style="text-align:center; padding: 8px;">Accuracy Guess 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center; padding: 8px;">Unfited Voter</td>
      <td style="text-align:center; padding: 8px;">78.2%</td>
      <td style="text-align:center; padding: 8px;">95.7%</td>
    </tr>
  </tbody>
</table>

<p align="center"> <img src="assets\fited_voter_conf_matrix.png" width="45%"> <img src="assets\fited_voter_conf_matrix_p2.png" width="45%"></p>


<div align="center" style="display: flex; justify-content: center; gap: 40px; align-items: flex-start; max-width: 1200px; margin: auto;">

  <table style="width: 45%; max-width: 600px; border-collapse: collapse; border: 1px solid #ccc;">
    <thead>
      <tr>
        <th style="text-align:left; border-bottom: 1px solid #ccc; padding: 8px;">Metric</th>
        <th style="text-align:center; border-bottom: 1px solid #ccc; padding: 8px;">Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left; padding: 8px;">Test Loss</td>
        <td style="text-align:center; padding: 8px;">0.5565</td>
      </tr>
      <tr>
        <td style="text-align:left; padding: 8px;">Params</td>
        <td style="text-align:center; padding: 8px;">17.7 K</td>
      </tr>
    </tbody>
  </table>

</div>
<br>



![Fited voter confusion matrix](assets/fited_voter_logs.png)


## Future Improvements
* Use more instances of classes 1,3,8 that exist in highway dataset 1.
* Use better ways of creating artificial 1,3,8 samples.








