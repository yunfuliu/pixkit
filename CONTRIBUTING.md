Online version please go<br>
<https://github.com/yunfuliu/pixkit/blob/master/CONTRIBUTING.md>

The initial form of CONTRIBUTING.txt comes from scikit-image <br>
<https://github.com/scikit-image/scikit-image>

Development process
===================

Here's the long and short of it:

1. If you are a first-time contributor:
---------------------------------------

   * Sign up for github (if you have no accounts): [Go here](https://github.com/)

   * Be a fork:
   
       Go to <[pixkit](https://github.com/yunfuliu/pixkit)>
       and click the "fork" button to create your own copy of the project
       (called ``origin`` remote).

   * Clone the project to your local computer:

       Download GitHub for your system from 
       <https://help.github.com/articles/set-up-git><br>
       Login and clone the project.

   * Add upstream repository:

       To keep track of the original repository, you need to add 
       another remote named ``upstream``.<br>
       
       Right click on the project after you clone it, and click "open a shell here"
       ``git remote add upstream https://github.com/yunfuliu/pixkit.git``

       Checkout by ``git remote``

   * Now, you have remote repositories named:

       - ``upstream``, which refers to the 'pixkit' repository
       - ``origin``, which refers to your personal fork <br>
       You can check by 
         ``git remote``

*NOTE* 

    About 'remote', please go <https://help.github.com/articles/fork-a-repo>
    for details.

2. Develop your contribution:
-----------------------------
Following operations are needed to keep track of the original repo.

   * Pull the latest changes from ``upstream``:

       ``git checkout master``<br>
       ``git pull upstream master``

   * Go to the GUI, and create a new branch for the feature you want 
     to work on. Since the branch name will appear in the merge message, 
     use a sensible name such as 'add-halftoning' (add funcationality of 
     halftoning):

*NOTE* 

    Please base upon the branch 'master' to create your new branch,
    rather than 'release'.
    Also, branch name 'add-???' for adding new functionality;
    branch name 'bug#???' for debuging. 

   * Commit locally as you progress (``git add`` and ``git commit``)

3. Work on your contributions:
-----------------------------

   * For adding new functionality:<br>
     1) Define your functions in `./pixkit-xxx/*.hpp`<br>
     2) Place your source code in `./pixkit-xxx/src/*.cpp` 
        Notably, please completely test your functions, such as various 
        image sizes.<br>
     3) Organize your example in `./examples-xxx` for each of your 
        functionality<br>
     4) Explane your functions in [wiki](https://github.com/yunfuliu/pixkit/wiki)

   * For debugging: 
     1) When you finish, please let us know during pulling requests. 

4. To submit your contribution:
-------------------------------

   * Push your changes back to your fork on GitHub::

      ``git push origin transform-speedups``

   * Go to GitHub. The new branch will show up with a Pull Request button -
     click it.

*NOTE*

    To reviewers: add a short explanation of what a branch did to the merge
    message and, if closing a bug, also add "Closes gh-123" where 123 is the
    bug number.


Divergence between 'upstream master' and your feature branch
--------------------------------------------------------------

Do *not* ever merge the main branch into yours. If GitHub indicates that the
branch of your Pull Request can no longer be merged automatically, rebase
onto master::

   ``git checkout master``<br>
   ``git pull upstream master``<br>
   ``git checkout transform-speedups``<br>
   ``git rebase master``

If any conflicts occur, fix the according files and continue::

   ``git add conflict-file1 conflict-file2``<br>
   ``git rebase --continue``

However, you should only rebase your own branches and must generally not
rebase any branch which you collaborate on with someone else.

Finally, you must push your rebased branch::

   ``git push --force origin transform-speedups``

(If you are curious, here's a further discussion on the
`dangers of rebasing <http://tinyurl.com/lll385>`__.
Also see this `LWN article <http://tinyurl.com/nqcbkj>`__.)


Guidelines
==========

* All code should be documented, to the wiki<br>
  <https://github.com/yunfuliu/pixkit/wiki>

* For new functionality, always add an example to the corresponding 
  <b>/examples-xx folder</b>.

* No changes should be committed without review. Ask 
  [Yun-Fu Liu](yunfuliu@gmail.com) if you get no response to your pull request.
**Never merge your own pull request.**

