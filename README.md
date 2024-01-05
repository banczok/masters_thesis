
# Opisy
- plots/* - export obrazow do png 
- model_compare - nerka 1 (na tej uczony model)
- model_compare_kidney_* - (test na innych ktorych model nigdy nie widzial)
- models.py - implementacja kilku modeli
- preprocess.ipynb - test normalizacji historgramu 
- reconstruction.ipynb oraz test_reconstruction.py - testy marching_cubes i export do stl
- train_test.ipynb - uczenie sieci 

# Co dalej
- patch do 512x512 z overlap
- architektura dpn oraz swissunetr
- wiecej epok (pewnie ok. 300 z early stopping po 50)
- nauka na wszystkich danych podzial 0.8 lub 0.85 na test/val
- wkorzystnie maszyny balic oraz google cloud 300usd

> Info 
> - modele teraz: 
	- resize do 512x512
	- uczenie tylko na kidney1_dense
	- tylko ok. 50 epok

# Obecne wyniki modeli
### kidney_1_dense
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>Dice Coefficient</th>
      <th>IoU</th>
      <th>Average Loss</th>
      <th>param_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>best_model_resnet50d_50epoch.pth</td>
      <td>0.927729</td>
      <td>0.869268</td>
      <td>0.999134</td>
      <td>0.892966</td>
      <td>0.816190</td>
      <td>0.002423</td>
      <td>53051080</td>
    </tr>
    <tr>
      <td>best_model_dpn92_50epoch.pth</td>
      <td>0.941729</td>
      <td>0.783217</td>
      <td>0.998936</td>
      <td>0.847248</td>
      <td>0.755072</td>
      <td>0.002797</td>
      <td>81201265</td>
    </tr>
    <tr>
      <td>best_model_regnetx120_45epoch.pth</td>
      <td>0.912151</td>
      <td>0.865844</td>
      <td>0.999064</td>
      <td>0.885466</td>
      <td>0.804429</td>
      <td>0.002625</td>
      <td>67260124</td>
    </tr>
    <tr>
      <td>best_model_res2net50_26w_8s_50epoch.pth</td>
      <td>0.913830</td>
      <td>0.859497</td>
      <td>0.999043</td>
      <td>0.880909</td>
      <td>0.798803</td>
      <td>0.002682</td>
      <td>73965920</td>
    </tr>
    <tr>
      <td>best_model_swissunetr.pth</td>
      <td>0.865630</td>
      <td>0.853083</td>
      <td>0.998734</td>
      <td>0.856807</td>
      <td>0.760868</td>
      <td>0.003595</td>
      <td>34337041</td>
    </tr>
    <tr>
      <td>best_model_unetr.pth</td>
      <td>0.822836</td>
      <td>0.625041</td>
      <td>0.997219</td>
      <td>0.700113</td>
      <td>0.550017</td>
      <td>0.007342</td>
      <td>133383521</td>
    </tr>
    <tr>
      <td>best_model_multiresunet.pth</td>
      <td>0.729220</td>
      <td>0.226819</td>
      <td>0.996032</td>
      <td>0.321402</td>
      <td>0.225678</td>
      <td>0.013273</td>
      <td>7251076</td>
    </tr>
    <tr>
      <td>best_model_resnet26d_150epoch.pth</td>
      <td>0.002688</td>
      <td>0.007871</td>
      <td>0.971877</td>
      <td>0.003781</td>
      <td>0.001896</td>
      <td>9.480476</td>
      <td>40497585</td>
    </tr>
    <tr>
      <td>best_model_unet3p.pth</td>
      <td>0.011189</td>
      <td>0.000339</td>
      <td>0.993802</td>
      <td>0.000655</td>
      <td>0.000329</td>
      <td>34.015163</td>
      <td>26923329</td>
    </tr>
    <tr>
      <td>best_modelR2AttU_Net.pth</td>
      <td>0.017969</td>
      <td>0.001518</td>
      <td>0.993331</td>
      <td>0.002777</td>
      <td>0.001395</td>
      <td>10.053096</td>
      <td>39442797</td>
    </tr>
  </tbody>
</table>
</div>
<br />

###  kidney_1_voi
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>Dice Coefficient</th>
      <th>IoU</th>
      <th>Average Loss</th>
      <th>param_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>best_model_resnet50d_50epoch.pth</td>
      <td>0.963721</td>
      <td>0.582376</td>
      <td>0.995000</td>
      <td>0.716416</td>
      <td>0.568598</td>
      <td>0.026464</td>
      <td>53051080</td>
    </tr>
    <tr>
      <td>best_model_dpn92_50epoch.pth</td>
      <td>0.969466</td>
      <td>0.588018</td>
      <td>0.995143</td>
      <td>0.722199</td>
      <td>0.576273</td>
      <td>0.030342</td>
      <td>81201265</td>
    </tr>
    <tr>
      <td>best_model_regnetx120_45epoch.pth</td>
      <td>0.970809</td>
      <td>0.518534</td>
      <td>0.994335</td>
      <td>0.663989</td>
      <td>0.510018</td>
      <td>0.051351</td>
      <td>67260124</td>
    </tr>
    <tr>
      <td>best_model_res2net50_26w_8s_50epoch.pth</td>
      <td>0.877661</td>
      <td>0.801725</td>
      <td>0.996721</td>
      <td>0.834914</td>
      <td>0.723326</td>
      <td>0.017846</td>
      <td>73965920</td>
    </tr>
    <tr>
      <td>best_model_swissunetr.pth</td>
      <td>0.561593</td>
      <td>0.578013</td>
      <td>0.990279</td>
      <td>0.559420</td>
      <td>0.396117</td>
      <td>0.154003</td>
      <td>34337041</td>
    </tr>
    <tr>
      <td>best_model_unetr.pth</td>
      <td>0.552456</td>
      <td>0.006921</td>
      <td>0.988546</td>
      <td>0.013649</td>
      <td>0.006880</td>
      <td>0.115405</td>
      <td>133383521</td>
    </tr>
    <tr>
      <td>best_model_multiresunet.pth</td>
      <td>0.873968</td>
      <td>0.082047</td>
      <td>0.989627</td>
      <td>0.136189</td>
      <td>0.081353</td>
      <td>0.058887</td>
      <td>7251076</td>
    </tr>
    <tr>
      <td>best_model_resnet26d_150epoch.pth</td>
      <td>0.011173</td>
      <td>0.119511</td>
      <td>0.884127</td>
      <td>0.020119</td>
      <td>0.010169</td>
      <td>52.362545</td>
      <td>40497585</td>
    </tr>
    <tr>
      <td>best_model_unet3p.pth</td>
      <td>0.403553</td>
      <td>0.022878</td>
      <td>0.988432</td>
      <td>0.042790</td>
      <td>0.022044</td>
      <td>93.864493</td>
      <td>26923329</td>
    </tr>
    <tr>
      <td>best_modelR2AttU_Net.pth</td>
      <td>0.067978</td>
      <td>0.016980</td>
      <td>0.986320</td>
      <td>0.026509</td>
      <td>0.013495</td>
      <td>25.437822</td>
      <td>39442797</td>
    </tr>
  </tbody>
</table>
</div>
<br />

###  kidney_2
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>Dice Coefficient</th>
      <th>IoU</th>
      <th>Average Loss</th>
      <th>param_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>best_model_resnet50d_50epoch.pth</td>
      <td>0.896928</td>
      <td>0.676580</td>
      <td>0.998388</td>
      <td>0.759228</td>
      <td>0.630437</td>
      <td>0.010688</td>
      <td>53051080</td>
    </tr>
    <tr>
      <td>best_model_dpn92_50epoch.pth</td>
      <td>0.863100</td>
      <td>0.669034</td>
      <td>0.998333</td>
      <td>0.743300</td>
      <td>0.606200</td>
      <td>0.008939</td>
      <td>81201265</td>
    </tr>
    <tr>
      <td>best_model_regnetx120_45epoch.pth</td>
      <td>0.857534</td>
      <td>0.740724</td>
      <td>0.998412</td>
      <td>0.780018</td>
      <td>0.657846</td>
      <td>0.012864</td>
      <td>67260124</td>
    </tr>
    <tr>
      <td>best_model_res2net50_26w_8s_50epoch.pth</td>
      <td>0.854820</td>
      <td>0.718850</td>
      <td>0.998435</td>
      <td>0.770868</td>
      <td>0.641942</td>
      <td>0.008089</td>
      <td>73965920</td>
    </tr>
    <tr>
      <td>best_model_swissunetr.pth</td>
      <td>0.620421</td>
      <td>0.687442</td>
      <td>0.997207</td>
      <td>0.624489</td>
      <td>0.472473</td>
      <td>0.015110</td>
      <td>34337041</td>
    </tr>
    <tr>
      <td>best_model_unetr.pth</td>
      <td>0.740983</td>
      <td>0.416340</td>
      <td>0.997125</td>
      <td>0.510367</td>
      <td>0.354028</td>
      <td>0.014349</td>
      <td>133383521</td>
    </tr>
    <tr>
      <td>best_model_multiresunet.pth</td>
      <td>0.869516</td>
      <td>0.210104</td>
      <td>0.997075</td>
      <td>0.325365</td>
      <td>0.209801</td>
      <td>0.012576</td>
      <td>7251076</td>
    </tr>
    <tr>
      <td>best_model_resnet26d_150epoch.pth</td>
      <td>0.010407</td>
      <td>0.091327</td>
      <td>0.970712</td>
      <td>0.017771</td>
      <td>0.008989</td>
      <td>7.624063</td>
      <td>40497585</td>
    </tr>
    <tr>
      <td>best_model_unet3p.pth</td>
      <td>0.234042</td>
      <td>0.040088</td>
      <td>0.996132</td>
      <td>0.065818</td>
      <td>0.037049</td>
      <td>17.071341</td>
      <td>26923329</td>
    </tr>
    <tr>
      <td>best_modelR2AttU_Net.pth</td>
      <td>0.027208</td>
      <td>0.012854</td>
      <td>0.994107</td>
      <td>0.016272</td>
      <td>0.008356</td>
      <td>8.365177</td>
      <td>39442797</td>
    </tr>
  </tbody>
</table>
</div>
<br />

###  kidney_3_sparse
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>Dice Coefficient</th>
      <th>IoU</th>
      <th>Average Loss</th>
      <th>param_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>best_model_resnet50d_50epoch.pth</td>
      <td>0.700456</td>
      <td>0.122708</td>
      <td>0.997610</td>
      <td>0.193128</td>
      <td>0.119886</td>
      <td>0.017400</td>
      <td>53051080</td>
    </tr>
    <tr>
      <td>best_model_dpn92_50epoch.pth</td>
      <td>0.011786</td>
      <td>0.000129</td>
      <td>0.997250</td>
      <td>0.000212</td>
      <td>0.000106</td>
      <td>0.026599</td>
      <td>81201265</td>
    </tr>
    <tr>
      <td>best_model_regnetx120_45epoch.pth</td>
      <td>0.624963</td>
      <td>0.113862</td>
      <td>0.997492</td>
      <td>0.183256</td>
      <td>0.108606</td>
      <td>0.018207</td>
      <td>67260124</td>
    </tr>
    <tr>
      <td>best_model_res2net50_26w_8s_50epoch.pth</td>
      <td>0.567939</td>
      <td>0.066090</td>
      <td>0.997392</td>
      <td>0.113085</td>
      <td>0.063409</td>
      <td>0.017672</td>
      <td>73965920</td>
    </tr>
    <tr>
      <td>best_model_swissunetr.pth</td>
      <td>0.230744</td>
      <td>0.039368</td>
      <td>0.997121</td>
      <td>0.064029</td>
      <td>0.035204</td>
      <td>0.025443</td>
      <td>34337041</td>
    </tr>
    <tr>
      <td>best_model_unetr.pth</td>
      <td>0.410216</td>
      <td>0.005579</td>
      <td>0.997314</td>
      <td>0.010751</td>
      <td>0.005542</td>
      <td>0.021404</td>
      <td>133383521</td>
    </tr>
    <tr>
      <td>best_model_multiresunet.pth</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.997315</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.019811</td>
      <td>7251076</td>
    </tr>
    <tr>
      <td>best_model_resnet26d_150epoch.pth</td>
      <td>0.000247</td>
      <td>0.002928</td>
      <td>0.962063</td>
      <td>0.000451</td>
      <td>0.000226</td>
      <td>11.287447</td>
      <td>40497585</td>
    </tr>
    <tr>
      <td>best_model_unet3p.pth</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.997075</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.643404</td>
      <td>26923329</td>
    </tr>
    <tr>
      <td>best_modelR2AttU_Net.pth</td>
      <td>0.000115</td>
      <td>0.000045</td>
      <td>0.995384</td>
      <td>0.000065</td>
      <td>0.000033</td>
      <td>6.194237</td>
      <td>39442797</td>
    </tr>
  </tbody>
</table>
</div>
