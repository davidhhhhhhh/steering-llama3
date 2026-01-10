## Llama3-70B-Instruct Experiment Result 
We grouped the tv_movie content based on age and sampled 100 content from each group as our pilot dataset. 

When prefixing prompt *repeat after me:*, the results are stored in  `refusal_eda/meta-llama_Meta-Llama-3-70B-Instruct_outputs_20251210_202004.csv`. 

We do not observe the model to refuse directly, except outputing nothing for certain content, and these *NA* responses have the following distributions: 
- R: 6
- PG-13: 5
- PG: 4
- G: 6

We also noticed above responses from models are poor quality, including containing repetition more than once. 

Therefore, we revised the prefix prompt to *repeat after me exactly once after the column:* and the results are stored in `refusal_eda/meta-llama_Meta-Llama-3-70B-Instruct_outputs_20260108_085044.csv`. 

For this time, we still did not observe any direct refusal, and the *NA* response have the following distributions among age groups: 
- R: 13
- PG-13: 9
- PG: 4
- G: 6

However, the responses are still poor quality. Specifically, for short content, it produces long and random responses before and after the original content; for long content, it only produces really short responses. We also notice it tends to start with word like *Meanwhile* which does not appear in the original content. 

Given our interests in modifying model's refusal behavior, we would replicaete the previous experiment from affine refusal paper and check whether the model would refuse anything at all. 
