?&$	?oA? @
???????FИI?K??!~nh?N?@$	}?(@??LtK????S4?=??!??
S?T@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?{,G???<K?P???A????????Y????Ұ?rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ګ??????g?????A???&???Y??????rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0E????8 @[??Ye???A0??L?^??Y1[?*±?rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0~nh?N?@W??m???A??m????Y*???
???rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0r?_!?	@^?????A?_ѭ???Y?????rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0'????@????	??A??r????Y\??.?u??rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0 ??	???????Q??A??c????Y????uo??rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?4?($?????????A+?gz????Y*??F????rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
˝?`8????D????A?n-??x??YE?<?\??rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?y????)&o?????A???????Y?ZH???rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?'??????d??J/??Ad:tz???YF;?I??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails03??3????T??E	??A????z??Y?#bJ$ѫ?rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ю~7???-?R\U??A????=z??Yl#???rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0FИI?K??}%??6??A????,???Y?z????rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~7ݲ??iUK:???A~T?~O??Y*r??9???rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0l&?ls???????A?8毐??Y?F ^?/??rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??\??????wJk??A?v0b? ??Y???ّ???rtrain 33*	????xқ@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???????!??7?:C@)j??????1{J<J??@@:Preprocessing2T
Iterator::Root::ParallelMapV2?Ԗ:????!?+mո[2@)?Ԗ:????1?+mո[2@:Preprocessing2E
Iterator::Root]k?SU???!???xA@)?Hm????1J
pc?0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice"r?z?f??!˷"??"@)"r?z?f??1˷"??"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??Ң>???!??CP@)????9??1???؉?"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate?V?9?m??!???9?,@)?v??????1?{5@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?*n?b~??!@????@)?*n?b~??1@????@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap"¿3??!?8f??1@)p[[x^*??1??1?ix
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?-1π?@I?v???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	ζ?F?e????fS?g???g?????!W??m???	!       "	!       *	!       2$	ŕ
??????$??????????,???!??m????:	!       B	!       J$	??]?P???S???ǰ?F;?I??!*???
???R	!       Z$	??]?P???S???ǰ?F;?I??!*???
???b	!       JCPU_ONLYY?-1π?@b q?v???X@Y      Y@q-=??-q?@"?	
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?31.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 