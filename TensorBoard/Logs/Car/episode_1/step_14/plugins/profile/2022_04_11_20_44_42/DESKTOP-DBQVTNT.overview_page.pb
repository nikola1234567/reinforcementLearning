?&$	$?e???@????????8?????!(????|@$	??	? ]
@ڭˍrN??????
???!e׾E@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????@??fG????AI??r????YY??w?'??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0-?}M??#[A?R??A??C???YG6u??rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Qԙ{? @??????A8?L???Y???XP??rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?d??)A??'0??m??A?&?????Y?ۃ?/??rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0f?s~?#@#??????AZ?xZ~`??YvoEb???rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?#ӡ?s???f?|????A?g?????Y?$???}??rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0X9??v?@?{,}???A~r 
&??Y?&?+?V??rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	ݵ?|??????#??S??A??a????Y@?P???rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?uq???;?ީ?{??A?E?Sw??Y<??fԬ?rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0 %vmow??C?O?}???A?×?"$??YKO?\??rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0o??@??????A˟o????Y??x?@e??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????R@?.???@A???Đ\??Y2q? ???rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??[????i??֦???AT?4??-??YHk:!t??rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??8?????ffffff??A??qnn??Yl[?? ???rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?qs*????}?
D??A??(#.???Y7Ou??p??rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0臭???@ѓ2?a??A/񝘵??YϺFˁ??rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0(????|@Oq???Ap???@Y4e??E??rtrain 33*	??? ?U?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatx?ܙ	??!Ys9HrC@)???9?r??1TxH?@@:Preprocessing2E
Iterator::Root<??kP??!??+???@@)k~??E}??1?J?/?0@:Preprocessing2T
Iterator::Root::ParallelMapV2??Ց#??!P?U?'?0@)??Ց#??1P?U?'?0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipr 
fL??!<8j.??P@)N`:?۠??1E'4;??#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSliceL??O????!4'u??H#@)L??O????14'u??H#@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?-?????!?B??_@)?-?????1?B??_@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::ConcatenateiW!?'???!R?Q??-@)
If????1=>?ܵ@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??)???!??(??1@)Q?B?y???1}?1,?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?!?ݪ?	@I????3X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	As???V??2?#?9???ffffff??!?.???@	!       "	!       *	!       2$	???????1?7?????E?Sw??!p???@:	!       B	!       J$	?0?s????B??????XP??!4e??E??R	!       Z$	?0?s????B??????XP??!4e??E??b	!       JCPU_ONLYY?!?ݪ?	@b q????3X@Y      Y@q??l?G?)@"?	
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?12.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 