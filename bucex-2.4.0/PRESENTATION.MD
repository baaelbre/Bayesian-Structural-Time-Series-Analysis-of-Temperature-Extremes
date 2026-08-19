Broad outline for the COMPSTAT presentation:

1) Develop and explain the complete methodology for one canonical extreme series, TXx.
2) Compare it carefully with the Gaetan–Grigoletto and Huerta–Sansó approaches.
3) Then generalize to all six temperature summaries using hierarchical componentwise SSVS.

1. INFERENCE COMPARISON
    Fit a simple location-driven model
    a) a la Huerta (single site MCMC) or Wei (particle filters)
    b) a la Gaetan (particle filters and smoothers, choosing process noise variance)
    c) FS non-centred PGAS
2. MODEL COMPARISON
    SSVS ...

bucex.model(....)

The important distinction is that Gaetan and Huerta smooth an already selected state specification. Your model also learns the specification.

3. MULTI-SERIES MODEL
    Fit a hierarchical model to all six series simultaneously, using componentwise SSVS to learn the specification of each series. This is a more general approach than the previous two, which only fit one series at a time.