?&$	?B??Y$@B?~?E????^?t??!>^H???@$	. ;??R@z???????5?dd???!fp???9@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?a???????7h????A??kCŸ??Y<O<g??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??LL??????̯&??A??????Y?:?p˯?rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???I?`@?*4˦??A[C???6??Y?q??????rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?b??*? @??4L???A&p?n????Y?G??[??rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?QcB??@P?I?5?@Ak,am????Y>?-z??rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0}ZEh&??!\?z:??A?Y?X???Y?%r????rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Z???? ??4????A??Χ???Y\?-?׭?rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	Q?????FaE???A?}?෡??Y/3l?????rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?SrN?!@?7? ???A?f?8f??Y??fF???rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?S??q@?-9(?@AN???x??Yw??/ݬ?rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0>^H???@R+L?kH??A?֊6?y??Y???????rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?zM

@??.5B???A????????Y??????rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0u?l???????J??F??A?Ù_???Y.=??????rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?T?t<???Gu:????A~SX??"??Y5??????rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0gC??A?????:?0??A????-??Y?\6:称?rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0kdWZFj @mscz????A?Az?"??YŎơ~??rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?^?t???!7?x??A??m?????Y*???O??rtrain 33*	?/ݤ??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat˻????!?I;?AC@)%?/???1?r???@@:Preprocessing2E
Iterator::Root"?4????!d?+o??@)P)????1???ڵ81@:Preprocessing2T
Iterator::Root::ParallelMapV2??Q??Z??!?P?xr_-@)??Q??Z??1?P?xr_-@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSliceϠ????!dN???'@)Ϡ????1dN???'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipӅX?F??!g	5:?Q@)tA}˜.??1g?[rь"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???????!L???g@)???????1L???g@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate???v??!V,?k_0@)a?$?Ӷ?1|?3?C@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??͋_??!6$?KM4@)_|?/???1)q6?o@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??=???@I??	?HX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?[Ѐ???g )?z???!7?x??!P?I?5?@	!       "	!       *	!       2$	??|9???(Ũ??R????m?????!?֊6?y??:	!       B	!       J$	X`{x??????????5??????!??????R	!       Z$	X`{x??????????5??????!??????b	!       JCPU_ONLYY??=???@b q??	?HX@Y      Y@q<? ??)@"?	
both?Your program is POTENTIALLY input-bound because 57.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?12.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 