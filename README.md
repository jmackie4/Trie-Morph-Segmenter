*UPDATE 2* - Holy cow, I finally finished all the main objects for this project. There's now an actual morphological segmenter that you can use. It takes in a list of words (strings) as an input in order to initialize the segmenter. Then it can be called on a list of words (strings) to segment them by adding a period where it thinks you should split the word into two morphemes. IT WILL PRODUCE SHORT SEGMENTATIONS!!! Right now, I'm just gonna be working on creating the full pipelines and cleaning up some of the code in this. Also the tests are fine, but I want to comeback to those and refine them later since they need to be improved!  


*UPDATE 1* - This project is still under construction, but it's almost finished! I've added the math_utils module that has everything you need 
in order to get the mutual information of two random variables from a joint frequency table! 

This project is still under construction! However, if you ever want to store a language's vocabulary within a trie, you can do it with the current iteration of the project.
Once the project is fully complete it'll serve as a morphological segmenter that uses the trie data structure to create the morpheme list!
For now though, if you just want to use this as a package, the main code is in the triemorph folder! A lot of the main items are actually scikit-learn estimator and transformer objects, so if your project uses sci-kit learn, then you'll be able to make use of the objects in the project! 



