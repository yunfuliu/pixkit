[Online version](https://github.com/yunfuliu/pixkit/blob/master/pixkit-image/readme.md)

For users
---------
* Include `pixkit-image.hpp`.
* Include the source code placed in `/src` you needed.<br>Corresponding examples please go [example](https://github.com/yunfuliu/pixkit/tree/master/examples-image).

For contributors
----------------
* Function definitions please place in `pixkit-image.hpp`.
* Place your source code in `/src`.<br>If your code needs 
  lots of functions required by only yourself, 
  you can use an isolate file for them with a name as short 
  as possible (example please see the NADD2013 case).<br> 
  Otherwise, we recommend you to implement your function
  into corresponding `/src/.cpp`. For instance, if you are
  implementing a filtering function, you can implement it 
  into `src/filtering.cpp`. 
