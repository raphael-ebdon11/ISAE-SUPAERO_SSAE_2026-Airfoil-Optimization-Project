Note: The current version of the optimizer was run using Python 3.14 and may not work on other version.

How to run:
1. Download the entire folder into IDE of your choice. This optimizer was developed in VS Code, so functionality cannot be confirmed for other IDEs.
2. Run airfoil_optimizer.py (the _xtr version uses forced separation for anyone who would like to experiment).
3. Let run. The optimizer will take several hours to test over 25,000 airfoils, so be patient.
4. Progress can be tracked in a runs folder which will appear with worker logs to see if optimization is working as intended.
5. Once completed, final results of the optimization will be displayed in the airfoil_results files.

Note: In this project specifically, results were compared between the final optimized foil and both the prelimary optimized foil (the program was run once to get a better seed for further optimization) and the Selig S5010 baseline airfoil.
