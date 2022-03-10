# DLST_parameter_design_via_GP_emulation
Dynamic Line Scan Thermography parameter design via Gaussian process emulation

We address the challenge of determining a valid set of parameters for a dynamic line scan thermography setup.
Traditionally, this optimalization process is a labour and time intensive work even for an expert skilled in the art.
Nowadays, simulations in software can reduce some of that burden.
However, when faced with many parameters to optimize, all of which cover a large range of values, this is still a time-consuming endeavour.
A large number of simulations are needed to adequately capture the underlying physical reality.
We propose to emulate the simulator by means of a Gaussian process. This statistical model serves as a surrogate for the simulations.
To some extent, this can be thought of as a “model of the model”.
Once trained on a relative low amount of data points, this surrogate model can be queried to answer various engineering design questions.
Moreover, the underlying model, a Gaussian process, is stochastic in nature.
This allows for uncertainty quantification in the outcomes of the queried model, which plays an important role in decision making or risk assessment.

