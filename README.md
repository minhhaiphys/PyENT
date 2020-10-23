# PyENT
Python version of FourmiLab's [ENT](http://www.fourmilab.ch/random/): Benchmarking suite for pseudorandom number sequence.

### Tests

* Entropy
* Chi-square
* Arithmetic mean versus Median
* Monte-Carlo value for Pi
* Serial correlation coefficient

### Usage

Convenient functions can be found in ```src/pyent.py```.

*Running from Terminal/Command line*

Run ```python pyent.py <list of file names>```. The program will read each file in binary format, convert the contents to an array of bytes then perform the tests.
