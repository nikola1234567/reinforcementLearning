?%$	??C^~$@???3???f????"@!+5{?'@$		??	???WU#m#??????_???!?;?Ίj??"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????&@(G?`F??A?2?F?$@Y?e??E??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1+5{?'@~;??"??A?Ov3+%@Y????9???r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1Ͻ?K?&@w?????Ax????$@Y2?m??f??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??u??e$@nN%@??A?k$	??"@Y???????r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1o?$??#@?ؖg)??A?C p?!@Y?.Ȗ???r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?H.?!?$@?m1??A??q9#@YG?g?u???r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1f????"@????????A??F??!@Y}?1Y???r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	??????#@??Z}u??A?|A	?"@Y???~??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
??z?"x#@K???>???A?67?'"@Yp??G7??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1H??0~r#@?z?2Q??Acc^G"@Y?8?????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1o?;2V[$@?3ڪ$???AUގpZ?"@Y"?:?v٧?r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1(?bd?t#@??cO??A???o^"@Y??~j?t??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails10?????#@??4F????A???ԱB"@Y???????r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??t?i?#@H?C??]??APn???"@Y]??$????r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?w?~?%@??s?f??A?)t^co$@YRcB?%U??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????$@?ꐛ????A=?K?e?"@Y???W??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?X???$@ݲC????A>v()H#@Y?u?;O<??r	train 517*	ʡE?s??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??R?h??!?'&??D@)?NGɫ??1???S??B@:Preprocessing2T
Iterator::Root::ParallelMapV2N4?s???!ʻ?56K2@)N4?s???1ʻ?56K2@:Preprocessing2E
Iterator::Rootq?i݆??!???t?!A@)??q4GV??1@??f??/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?ܵ?|???!%??E?oP@).??:???1g??M.?#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?? ???!?ϓۣ?#@)?? ???1?ϓۣ?#@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??\???!%?Yqbj@)??\???1%?Yqbj@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??]?????!?eW],@)???@?M??1????r?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?<Oԁ???I?aW???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	60X??+[??߹?????????!w?????	!       "	!       *	!       2$	??74?"@?e"s|~????F??!@!?Ov3+%@:	!       B	!       J$	v@??E???Y???Gt??u?;O<??!]??$????R	!       Z$	v@??E???Y???Gt??u?;O<??!]??$????b	!       JCPU_ONLYY?<Oԁ???b q?aW???X@Y      Y@q?J??????"?
both?Your program is POTENTIALLY input-bound because 7.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 