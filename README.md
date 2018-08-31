# Support-Vector-Machine
Support vector machine (SVM) is a supervised learning model used for classification.

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
We want to find decision boundary <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctextbf%7B%5Ctextit%7Bw%7D%7D%5E%5Ctextbf%7B%5Ctextit%7Bt%7D%7D%5Ctextbf%7B%5Ctextit%7Bx%7D%7D&plus;b%3D0" /> whose margin is maximized for training set <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5C%7B%28%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%2Cy_i%29%5C%7D%24%2C%20%24%20y_i%3D&plus;1/-1" /> 
</p>
