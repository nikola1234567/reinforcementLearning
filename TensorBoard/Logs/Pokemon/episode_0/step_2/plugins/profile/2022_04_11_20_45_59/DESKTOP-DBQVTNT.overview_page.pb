?*$	S)?!4T@??9????b?? ????!F?T?=?@$	~8#?n?@??y He???ay:I??!g?hr?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??????@I?,|}-??A? OZ?l??Yܝ??.4??rtrain 53"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0F?T?=?@?o^??j @A~T?~Ol@YU??X6s??rtrain 54"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??:r??
@.?Ue????A??Z??W??Y?P?,???rtrain 55"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?߾??@KxB??	??AE???JY??Yp[[x^*??rtrain 0"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/¦Σ?@8?0????A?7N
????Y???9????rtrain 1"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/Gq?::?@?$??}x??A,F]k????Y??~? ??rtrain 2"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/lA??!?@jj?Z???AS???җ??Y<hv?[???rtrain 3"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/	??(?'?@?&p????A&??????YM??u??rtrain 4"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/
???#*@Tt$?????A?????Y?0????rtrain 5"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/????m???h?????A???U????YJ?E???rtrain 6"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/??'??@???????A??_Yi???Y9??m4???rtrain 7"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/??4?ׂ??m?Yg|???A* ?????YZd;?O??rtrain 8"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?v/?ɑ??e?pu D??A?.l?V???YO@a?ӫ?rtrain 9"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0b?? ?????wD???AZd;?O???Y???#ӡ??rtrain 10"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0˂???z @r2q? ???A?N?z1??Y?tu?b???rtrain 11"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?f?baH??n???k??A??,?????Y7?[ A??rtrain 12"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????7?n?e??Aq????i??Y? ??	L??rtrain 13*	?&1?ʣ@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?P??ќ??!???45C@)Uh ????1#??VA@:Preprocessing2E
Iterator::Rootn1?74e??!I ?2d:@)??[?nK??1>Z?+@:Preprocessing2T
Iterator::Root::ParallelMapV2??"??~??!U??JH)@)??"??~??1U??JH)@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapHk:!t??!Zh??V*2@)
?2???1?????&@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???????!U܍?T"@)>?*??1/??v?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???uR??!???_P@)???
???1?IR0?"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice}y?ѩ??!O?jP+A@)}y?ѩ??1O?jP+A@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*lA??! ??!?L?"?D@)lA??! ??1?L?"?D@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat噗??;??!?a\??l??)?[[%X??1U̯?2??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceqW?"???!???i8??)qW?"???1???i8??:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchU?]=??!8??Ϸ??)U?]=??18??Ϸ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range ?C??<n?!??d?????) ?C??<n?1??d?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9:??j@I@/?z?TX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??????O.???7?n?e??!?o^??j @	!       "	!       *	!       2$	?۫?????*?\N??Zd;?O???!~T?~Ol@:	!       B	!       J$	??oIj???߄????0????!U??X6s??R	!       Z$	??oIj???߄????0????!U??X6s??b	!       JCPU_ONLYY:??j@b q@/?z?TX@Y      Y@q$?T?BB@"?	
both?Your program is POTENTIALLY input-bound because 55.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?36.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 