?&$	??wJL???1?dA??Q?+?O??!aTR'?I@$	?????@	'?p?4??μ?<??!?j1hܬ@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0v?|?H???	?/?
??A?B,cC??Y? v??y??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?4`??)@??????A?ꭁ???Y@?? kղ?rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0N??1m@???s?@??A)x
?R??Y?Sͬ???rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?y7??<O<g???A???[??Y??5???rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0R?d=???? 4J?~??A?v?k?F??Y?4S??rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D???????jO?9???A㥛? ???Yߩ?{????rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails03ı.n# @[Υ??l??A?!??gx??Y4f????rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	an?r?|
@??'?(@A?f*?#???YGT?n.???rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
aTR'?I@?n??\???Az??C5e??Y稣?j??rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails00??L??@???`?#@A??????Y??B?iީ?rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Du??w?Ny4??A+??X????YQ?Hm??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?|\*???H?}8???A(??Z&???Y???ګ??rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??>?Q^??EJ?y??A?m?(??Y?n?燩?rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails07??­??????9??Aђ?????Y*??????rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D4??ؙ??<g???A?$y????YV?j-?B??rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Y?%??R~R?????A ?={.S??Y???tx??rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Q?+?O??j????g??A??????Y)?A&9??rtrain 33*	?(\???@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatׇ?F?p??!??????H@)j?TQ<??1?d??IF@:Preprocessing2T
Iterator::Root::ParallelMapV2?HJzZ??!?qT?0*@)?HJzZ??1?qT?0*@:Preprocessing2E
Iterator::Root?&?W??!???u+-:@):?6U??1E????)*@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????h???!??"?tR@)?`R||B??1???N~O!@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSliceF?v???!?????!@)F?v???1?????!@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?,%?I(??!Eۋ?$?@)?,%?I(??1Eۋ?$?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenatei??r????!ڡQ?<))@)? ???1??M?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapʋL?????!??? ?.@)#/kb???1o7pf{@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9=mH?b?	@I??m??0X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?(?̡??aOC?n??<g???!??'?(@	!       "	!       *	!       2$	?`?N??swY#??ђ?????!z??C5e??:	!       B	!       J$	i0???3??<???????n?燩?!?Sͬ???R	!       Z$	i0???3??<???????n?燩?!?Sͬ???b	!       JCPU_ONLYY=mH?b?	@b q??m??0X@Y      Y@qu??+q4@"?	
both?Your program is POTENTIALLY input-bound because 59.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?20.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 