?&$	N? @?ᆔ?U??ҏ?S????!?????	@$	??}D??	@??ɍ
?????\??^ @!ϯ?,@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails05???#???Z??c!???A?=z?}d??Y$`tys???rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?πz3*@0e?????A???=????Y
?]?V??rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Hj?dr??????I'???A?v|????Yy@ٔ+???rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails05]Ot]?@C???????A??Ü ??Y?ʅʿ???rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?9$?Pr@5D?/??A8????v??Y??ND??rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????	@?"??JV @A?º?.??YwJ????rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0K#f?y?	@4?"1???A?!yv??Y	??8?d??rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	g'??? @+???+^??A???s??Y????);??rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
????5???j?t???A???y???Y????q¬?rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0e???\???-y??A1?Tm7A??Y????}r??rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?eM,?? @??kC??AX:?%???Y?8?ߡ(??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?K?'7??v?1<???A???=???Yb???°?rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0l#?????.??????AԞ?sb??Yݗ3????rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails00?'u??o??o????Aa??????Y???;???rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0rP?Lۿ??tѐ?(???A*?dq????Y?^zo??rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0(F??1??~? ????A??3w???Y??cw????rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ҏ?S????OI?V??A?jQL???Y?p?Ws???rtrain 33*	??n?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatN?t"???!#???C@)<?H?????1??b???@@:Preprocessing2E
Iterator::Root???I????!i)??kA@)????ɍ??1X1*Z?2@:Preprocessing2T
Iterator::Root::ParallelMapV2???k???!z(???0@)???k???1z(???0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice>\r?)??!L6???>$@)>\r?)??1L6???>$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipw?
??f??!?q??/JP@)VIddY??1?????"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?Ȱ?72??!Cyj??y@)?Ȱ?72??1Cyj??y@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate??ܚt[??!?ȝu%-@)f???~3??1!щ"??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??:ǀ???![?@KS?1@)?f?ba???1e?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?k???@I?4?2W:X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?5)a????%???g??OI?V??!?"??JV @	!       "	!       *	!       2$	?g??D???'|?,??*?dq????!?!yv??:	!       B	!       J$	QYfcwӯ?C?m??W???p?Ws???!	??8?d??R	!       Z$	QYfcwӯ?C?m??W???p?Ws???!	??8?d??b	!       JCPU_ONLYY?k???@b q?4?2W:X@Y      Y@qZ???+3@"?	
both?Your program is POTENTIALLY input-bound because 60.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?19.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 