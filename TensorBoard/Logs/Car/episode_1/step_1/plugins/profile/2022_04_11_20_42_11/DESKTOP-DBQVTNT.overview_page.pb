?&$	(??]???"N`??B??c'????!'g(?x#@$	?:??0@?)tc/???|??A@!?ƣը@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????\?z????Az?rK+??Yg?lt?O??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?̱?????J&?v????A?L?????YH?`?ڭ?rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??????????0a4+??A׾?^????Ym???e??rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0F\ ?K??s??=Ab??A?]???T??Y?6qr?C??rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0'g(?x#@>?ɋ???A?F?0}?@Y??`??rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0>??@?7?ܘ???Ai?hs?[??Y?YL??rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ۤ??v@??g?????A???I?:??Y???g????rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?Q???4 @??YKi??AJ&?v?)??Ya?.?e???rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
h????????%?L???Ar?_!???Y??릔ת?rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?d????qs*Y??A?w??????Y>	l??3??rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0\???4L???? ?c??Ae??]? ??Y?!S>??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??	h"l??[?? ????A?c?????Ye?fb??rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?|? @?u???_??An4??@??Y%@M-[???rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?R???????ڧ?1???A~?????Y?3???l??rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0c'?????9? ???AQMI?????Y[]N	?I??rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0t?a??0)>>!???A?
?r???Y]?wb֋??rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?R@??@???
b?k???A@??
/??YZ???а??rtrain 33*	?t??\?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??|?X???!?A[܋E@)???T?(??1[?iT?A@:Preprocessing2E
Iterator::Root`??V????!c??MNk:@)??	K<???1?Bx?ј+@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-????)??!??l,eR@)pxADj???1?ۼ??*@:Preprocessing2T
Iterator::Root::ParallelMapV23??bb???!7?'??=)@)3??bb???17?'??=)@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice??{?????!\??"@)??{?????1\??"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*/l?V^???!^??VB?@)/l?V^???1^??VB?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenatem?Yg|_??!?????=+@)??n,(??1bP?i5P@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapK"? ˂??!??>?;1@)wۅ?:???1????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???`??@I??D{X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	K??u????#V+?J????9? ???!>?ɋ???	!       "	!       *	!       2$	?G/	????v?@?\??@??
/??!?F?0}?@:	!       B	!       J$	???f????[??l?T??[]N	?I??!??`??R	!       Z$	???f????[??l?T??[]N	?I??!??`??b	!       JCPU_ONLYY???`??@b q??D{X@Y      Y@q? ?G3@"?	
both?Your program is POTENTIALLY input-bound because 54.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?19.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 