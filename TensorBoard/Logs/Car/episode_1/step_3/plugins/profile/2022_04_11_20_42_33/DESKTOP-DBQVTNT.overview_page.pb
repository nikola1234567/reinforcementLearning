?&$	?~ @?T?:?j??i??	??!]?mO??@$	??u?m?@???o? ?????/*???!3?&6?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?@??????'????ASy=????Y?}͑??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0)?k{@ 8???L@AA?v???Y=?N?P??rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Uס??@Yj??h???A???????Y?:????rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0+Kt?Y????lXSY??A+O ?+??Y7???N???rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????{?%T??A??g͏???Y?E????rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?vj.7???y>??(??AuXᖏ??Y|??????rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?L??? @	5C?(???A?1?????Y?~?d?p??rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	K??z2? @?:?????A|E?^Ӄ??Y?衶??rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
tѐ?(?@P??H?\@A%????g??Y?6?Ӂ???rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0]?mO??@??d?`?@A????K~??Y??M(??rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0i??	???7k?????A???B????Y>?Ӟ?s??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0U?=ϟ6??9	?/???A????9??Y?n?o?>??rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0N)????????????A????????Y?;? ??rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ѓ2??- @:=??B??A^???T{??Y?Z?kBZ??rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??????Z? m???A?ߠ?????Y??b?d??rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0"T??-????#?????A'?_???Y? ??*???rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Y?????p???????A\u?)???Y??{h??rtrain 33*	ףp=o?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat????????!??f6 ?D@)8? "5???1 ?9* ?@@:Preprocessing2E
Iterator::Root???????!??? ??>@)?;P?<???1???/@:Preprocessing2T
Iterator::Root::ParallelMapV2????E??!???HP.@)????E??1???HP.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?:?????!PK???SQ@)??.ޏ???1V%???#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice{???ȭ??!Aj?)1?!@){???ȭ??1Aj?)1?!@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??DKO??!0g?0  @)??DKO??10g?0  @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenateˁj?0??!?pTs!-.@)6l??g??1D`??c@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????t ??!???NZ2@)D1y?|??1J?Rw?
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?H9?;-@I?5& ?&X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	~?-X???????w*???7k?????!P??H?\@	!       "	!       *	!       2$	fq??
???+?	???????9??!????K~??:	!       B	!       J$	<?Q?????
?TVN???;? ??!??M(??R	!       Z$	<?Q?????
?TVN???;? ??!??M(??b	!       JCPU_ONLYY?H9?;-@b q?5& ?&X@Y      Y@q?R@X?5@"?	
both?Your program is POTENTIALLY input-bound because 62.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?21.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 