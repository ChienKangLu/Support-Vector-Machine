# Support-Vector-Machine
Support vector machine (SVM) is a supervised learning model used for classification.

## Develop tools and techniques
+ Python
+ Pycharm

## Algorithm
The core idea of SVM is to find a decision boundary which can seperate different class.
<table>
  <tr>
    <td><img height=250 src="https://github.com/ChienKangLu/Support-Vector-Machine/blob/master/img/pic.png" /></td>
    <td  valign="top">
        <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bll%7D%201.%20%26%20%24find%20%24%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%5C%5C%20%26%20%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Blr%7D%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%3D1%20%5Ccdots%20%5Ctextcircled%201%20%5C%5C%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%3D-1%20%5Ccdots%20%5Ctextcircled%202%20%5Cend%7Barray%7D%5Cright.%5C%5C%20%26%20%5Ctextcircled%201%20-%20%5Ctextcircled%202%20%5C%5C%20%26%20%28%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B1%7D%7D-%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B2%7D%7D%29%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%3D2%20%5C%5C%20%26%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%3D%20%5Cfrac%7B2%7D%7B%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B1%7D%7D-%20%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B2%7D%7D%7D%20%5Cend%7Barray%7D" />
    </td>
    <td valign="top">
      <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bll%7D%202.%20%26%20%24find%20%24m%24%20by%20projection%24%20%5C%5C%20%26%20%5Cbegin%7Barray%7D%7Blllll%7D%20m%20%26%20%3D%20%26%20proj_%7B%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B3%7D%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B3%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%7B%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%5C%5C%20%26%26%26%3D%20%26%5Cfrac%7B%28%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B1%7D%7D-%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B2%7D%7D%29%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%7B%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%5C%5C%20%26%26%26%3D%20%26%5Cfrac%7B%28%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B1%7D%7D-%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B2%7D%7D%29%5Ccdot%5Cfrac%7B2%7D%7B%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B1%7D%7D-%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7B2%7D%7D%7D%7D%7B%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%5C%5C%20%26%26%26%3D%26%5Cfrac%7B2%7D%7B%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5Ccdot%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%5C%5C%20%26%26%26%3D%26%5Cfrac%7B2%7D%7B%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%5Cright%20%5C%7C%7D%20%5Cend%7Barray%7D%20%5Cend%7Barray%7D" />
    </td>
  </tr>
</table>
<br/>
<p>
We want to find decision boundary <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D&plus;b%3D0" /> whose margin <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B2%7D%7B%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%5Cright%20%5C%7C%7D" /> is maximized for training set <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5C%7B%28%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%2Cy_i%29%5C%7D%24%2C%20%24%20y_i%3D&plus;1/-1" />.<br/>
At beginning, we will trasnform maximizing margin <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B2%7D%7B%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%20%5Cright%20%5C%7C%7D" /> to minimizing <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20J%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%29%3D%5Cfrac%7B1%7D%7B2%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D" /> with constraint <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20y_i%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%29%5Cgeq%201" />, all class of training data should be 1 or -1,
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Blr%7D%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%5Cgeq%201%20%24%2C%20if%20%24y_i%3D1%20%5C%5C%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%5Cleq%20-1%20%24%2C%20if%20%24y_i%3D-1%20%5Cend%7Barray%7D%5Cright." />
</p>
</p>
<p>
  Use Lagrange function (primary Lagrangians) to solve,
</p>
<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?L%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%2Cb%2C%5Cmathit%7B%5Cboldsymbol%7B%5Calpha%7D%7D%29%3D%5Cfrac%7B1%7D%7B2%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_i%5C%7By_i%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%29-1%5C%7D" />
</p>
<p>
Take the derivative of <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20L%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%2Cb%2C%5Cmathit%7B%5Cboldsymbol%7B%5Calpha%7D%7D%29" /> with respect to <b><i>w</i></b> and <i>b</i> and set to zero,
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bll%7D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%3D0%26%24%2C%20%24%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_iy_i%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%7D%3D0%26%24%2C%20%24%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_iy_i%3D0%20%5C%5C%20%5Cend%7Barray%7D" />
</p>
<p>
  Because the Lagrange multipliers are unknown, we still can not slove <b><i>w</i></b> and <i>b</i></u>. Lagrange multipliers for equality constraints are free parameters that can take any values. Therefore, we add the Karush–Kuhn–Tucker(KKT) conditions which constraint the Lagrange multipliers to be non-negative:
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bll%7D%20%5Cboldsymbol%7B%5Cmathit%7B%5Calpha%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%5Cgeq%200%20%5C%5C%20%5Cboldsymbol%7B%5Cmathit%7B%5Calpha%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%5By_i%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%29-1%5D%3D0%20%5Cend%7Barray%7D" />
</p>
<p>
  The constraint states the Lagrange multiplier <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathit%7B%5Calpha%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D" /> must be zero unless the training instance <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D" /> satisfies the equation <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20y_i%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%29%3D1" /> becasue <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathit%7B%5Calpha%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D" /> must be larger than or equal to zero. Moreover, these training instance whose <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathit%7B%5Calpha%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D" /> is larger than 0 is known as support vector. Also, only the support vectors define the decision boundary.
</p>
<p>
  For simplifying, we will transform the problem into a function of the Lagrange multipliers only (dual problem):<br/>
  Combine <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%7D%3D0" /> and <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%7D%3D0" /> into <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20L%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%2Cb%2C%5Cboldsymbol%7B%5Calpha%7D%29" />:
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Blll%7D%20L%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%2Cb%2C%5Cboldsymbol%7B%5Calpha%7D%29%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B2%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_i%5C%7By_i%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D&plus;b%29-1%5C%7D%20%5C%5C%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B2%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_iy_i%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_iy_ib&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_i%20%5C%5C%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B2%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D-%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_i%20%5C%5C%20%26%20%3D%20%26%20-%5Cfrac%7B1%7D%7B2%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_i%20%5C%5C%20%26%20%3D%20%26%20-%5Cfrac%7B1%7D%7B2%7D%5Csum%5Cnolimits_%7Bi%2Cj%7Dy_iy_j%5Calpha_i%5Calpha_j%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bj%7D%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_i%20%5Cend%7Barray%7D" />
</p>
<p>
  Find <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathbf%7B%7D%5Calpha%7D" /> that maximizes <img src="https://latex.codecogs.com/svg.latex?L%28%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%29" /> subject to <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7By%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%3D0" /> and <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%5Cgeq%200" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Blll%7D%20L%28%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%29%26%3D%26-%5Cfrac%7B1%7D%7B2%7D%5Csum%5Cnolimits_%7Bi%2Cj%7Dy_iy_j%5Calpha_i%5Calpha_j%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bj%7D%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_i%20%5C%5C%20%26%3D%26-%5Cfrac%7B1%7D%7B2%7D%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7DH%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D&plus;%5Ctextbf%7B%5Ctextit%7Bf%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%20%5C%5C%20H_i_j%26%5Cequiv%20%26%20y_iy_j%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D%5Ctextbf%7B%5Ctextit%7Bj%7D%7D%5C%5C%20%5Ctextbf%7B%5Ctextit%7Bf%7D%7D%20%26%20%3D%20%26%281%2C1%2C%5Ccdots%2C1%29%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%20%5Cend%7Barray%7D" />
</p>
<p>
  Take the derivative of <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20L%28%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%29" /> with respect to <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D" />:
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbigtriangledown_%7B%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%7DL%5Cequiv%20%5B%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Calpha_1%7D%2C%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Calpha_2%7D%2C%5Ccdots%2C%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Calpha_N%7D%5D%3D%5Ctextbf%7B%5Ctextit%7Bf%7D%7D-H%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D" />
</p>
<p>
  <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_k" /> must be decided by <i>N</i>-1 <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_i" /> because of <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7By%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%3D0" />:
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bll%7D%20%5Calpha_ky_k&plus;%5Csum%20%5Cnolimits_%7Bi%5Cneq%20k%7D%5Calpha_iy_i%20%3D%200%5C%5C%20%5Calpha_k%3D%5Cfrac%7B-1%7D%7By_k%7D%5Csum%20%5Cnolimits_%7Bi%5Cneq%20k%7D%5Calpha_iy_i%20%5Cend%7Barray%7D" />
</p>
<p>
  The gradient of <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Calpha_i%7D" /> is composed of two parts which are with respect to <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_i" /> and <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_k" />:
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bllll%7D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Calpha_i%7D%20%26%20%3D%20%26%20%28%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%7D%29_i%20&plus;%28%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%7D%29_k%5Cfrac%7B%5Cpartial%20%5Calpha_k%7D%7B%5Cpartial%20%5Calpha_i%7D%20%26%20%5Cforall%20i%5Cneq%20k%5C%5C%20%26%20%3D%20%26%20%28%5Ctextbf%7B%5Ctextit%7Bf%7D%7D-H%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%29_i%20&plus;%20%28%5Ctextbf%7B%5Ctextit%7Bf%7D%7D-H%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%29_k%5Cfrac%7B%5Cpartial%20%5Calpha_k%7D%7B%5Cpartial%20%5Calpha_i%7D%20%26%20%5Cforall%20i%5Cneq%20k%24%20%24%28%5Calpha_k%3D%5Cfrac%7B-1%7D%7By_k%7D%5Csum%20%5Cnolimits_%7Bi%5Cneq%20k%7D%5Calpha_iy_i%29%20%5C%5C%20%26%20%3D%20%26%20%28%5Ctextbf%7B%5Ctextit%7Bf%7D%7D-H%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%29_i%20&plus;%20%28%5Ctextbf%7B%5Ctextit%7Bf%7D%7D-H%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D%29_k%28%5Cfrac%7B-y_i%7D%7By_k%7D%29%20%26%20%5Cend%7Barray%7D" />
</p>
<p>
  Use gradient decsent to find <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_i" /> and <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_k" />, Let <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7Bd%7D%7D%5Cequiv%20%5Ctextbf%7B%5Ctextit%7Bf%7D%7D-H%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bll%7D%20%5Calpha_i%5E%7B%27%7D%3D%5Calpha_i&plus;%5Cvarepsilon%20%5Bd_i&plus;d_k%28%5Cfrac%7B-y_i%7D%7By_k%7D%29%5D%24%20%24%5Cforall%20i%5Cneq%20k%20%5C%5C%20%5Calpha_k%5E%7B%27%7D%3D%5Cfrac%7B-1%7D%7By_k%7D%5Csum%20%5Cnolimits_%7Bi%5Cneq%20k%7D%5Calpha_i%5E%7B%27%7Dy_i%20%5Cend%7Barray%7D" />
</p>
<p>
  Once we get <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D" />, we can use <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cboldsymbol%7B%5Cmathbf%7B%5Calpha%7D%7D" /> to calculate <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7B*%7D%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_iy_i%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D" />
</p>
<p>
  For calculating <i>b</i>, we use <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20y_i%28%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bw%7D%7D&plus;b%29%3D1" /> to obtain <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20b_i" /> for each support vectors and then average these values:
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bl%7D%20b_i%3D%5Cfrac%7B1%7D%7By_i%7D-%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%20%5C%5C%20b%5E*%3D%5Cfrac%7B1%7D%7BN_s%7D%5Csum%20%5Cnolimits_s%20%28%5Cfrac%7B1%7D%7By_s%7D-%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bs%7D%7D%29%20%5Cend%7Barray%7D" />
</p>

## Training detail
+ <img
src="https://camo.githubusercontent.com/dc7780e207c0060099d84a8672195eaeeb8d5215/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543696e6c696e65253230253543616c7068615f69" data-canonical-src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_i" style="max-width:100%;" /> must be within (0,<i>C</i>)

+ At each iteration, we will choose a survived <img src="https://camo.githubusercontent.com/f4e1dea119b29cde2faa516fc2d0858bf4376862/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543696e6c696e65253230253543616c7068615f6b" data-canonical-src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha_k" style="max-width:100%;"> as the dependent variable

## Practice
+ Number of class label: 2
+ Number of data: 100
+ <i>C</i>: 30
+ Learning rate: 0.005
+ Iteration: 100
+ Variation of loss 
  <table>
    <tr>
      <td><img height="200" src="https://github.com/ChienKangLu/Support-Vector-Machine/blob/master/img/loss_maximization.png" /></td>
      <td><img height="200" src="https://github.com/ChienKangLu/Support-Vector-Machine/blob/master/img/loss_maximization_0_12.png" /></td>
      <td><img height="200" src="https://github.com/ChienKangLu/Support-Vector-Machine/blob/master/img/loss_maximization_12.png" /></td>
    </tr>
    <tr align="center">
      <td>iteration 0 ~ 99</td>
      <td>iteration 0 ~ 12</td>
      <td>iteration 12 ~ 99</td>
    </tr>
  </table>
  
+ Result
  + There are 3 support vectors
  + <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20w%3D%5B-0.15040408%2C0.32960551%5D" />
  + <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20b%3D-2.33089085944" />

<p align="center">
<img height="250" src="https://github.com/ChienKangLu/Support-Vector-Machine/blob/master/img/result.png" />
</p>

## Reference
+ AI course of international management department in NTUST
+ Introduction to data mining---by Pan-ning Tan, Mich. Steinbach and Vipin kumar
