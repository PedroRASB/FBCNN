This repository contains the code for the paper "FBDNN: Filter Banks and Deep Neural Networks for Portable and Fast Brain-Computer Interfaces". 

Paper: https://doi.org/10.1088/2057-1976/ac6300

Preprint: https://doi.org/10.48550/arXiv.2109.02165

# ABSTRACT
Objective: To propose novel SSVEP classification methodologies using deep neural networks (DNNs) and improve performances in single-channel and user-independent brain-computer interfaces (BCIs) with small data lengths. Approach: We propose the utilization of filter banks (creating sub-band components of the EEG signal) in conjunction with DNNs. In this context, we created three different models: a recurrent neural network (FBRNN) analyzing the time domain, a 2D convolutional neural network (FBCNN-2D) processing complex spectrum features and a 3D convolutional neural network (FBCNN-3D) analyzing complex spectrograms, which we introduce in this study as possible input for SSVEP classification. We tested our neural networks on three open datasets and conceived them so as not to require calibration from the final user, simulating a user-independent BCI. Results: The DNNs with the filter banks surpassed the accuracy of similar networks without this preprocessing step by considerable margins, and they outperformed common SSVEP classification methods (SVM and FBCCA) by even higher margins. Conclusion and significance: Filter banks allow different types of deep neural networks to more efficiently analyze the harmonic components of SSVEP. Complex spectrograms carry more information than complex spectrum features and the magnitude spectrum, allowing the FBCNN-3D to surpass the other CNNs. The performances obtained in the challenging classification problems indicates a strong potential for the construction of portable, economical, fast and low-latency BCIs.

# CONTENT
The .py files have the code for the neural networks and filter banks.

TrainedModels folder contains the networks trained on the entire Benckmark dataset.

## Citation:
@article{Bassi_2022,
	doi = {10.1088/2057-1976/ac6300},
	url = {https://doi.org/10.1088/2057-1976/ac6300},
	year = 2022,
	month = {apr},
	publisher = {{IOP} Publishing},
	volume = {8},
	number = {3},
	pages = {035018},
	author = {Pedro R A S Bassi and Romis Attux},
	title = {{FBDNN}: filter banks and deep neural networks for portable and fast brain-computer interfaces},
	journal = {Biomedical Physics {\&}amp$\mathsemicolon$ Engineering Express}
}

## Contact:
Please contact pedro.salvadorbassi2@unibo.it for more information.
