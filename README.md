# K-Closest-Circles (K-means variation)


<h3>K-Closest-Circles Algorithm</h3>

<b>Input:</b><br>
  1. N = number of points
  2. X = the Npoints in the 2D coordinate system
  3. K = number of circles to fit
  
<b>Output:</b><br>
  1. P<sub>i</sub>, i &#8712; {1..K}: the assignment of points in X to the i-th circle (hence, sum(count(P<sub>i</sub>)) = N)
  2. C<sub>i</sub>, i &#8712; {1..K}: the circles, represented as centre coordinates and radii
  
<b>Algorithm Steps:</b><br>
  1. Create the set C of initial circles (C<sub>i</sub>, i &#8712; {1..K}) 
  2. While not finished:
    a. For each point, assign it to the circle that is closest to it - this populates all P<sub>i</sub>, i &#8712; {1..K}
    b. For every P<sub>i</sub>, fit a circle using least squares method - this creates a better set C
    
<b>Stopping Criteria:</b><br>
Stop if either of these conditions are met:
  1. No change is assignment of points to circles from the previous iteration 
  2. Maximum number of iterations is reached
    
    
    
