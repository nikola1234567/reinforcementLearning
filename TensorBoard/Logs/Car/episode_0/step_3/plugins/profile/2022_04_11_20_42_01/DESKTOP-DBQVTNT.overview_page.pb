?&$	?I??Gp??????f??┹?F???!:???
@$	?NW?B&@?;lS?????S%< @!??DS?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G 7?K@0?GQgn??AzT?????Y+???ڧ??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0:???
@'?5????A???K. @Y?ECƣT??rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???z???j???????A?A?Ѫ??Y?lXSY??rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?b????????R{???A?Nt???Ye?P3????rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Q?0???s???i??A!W?Y???Y??]gE??rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????????????A<?)??Y?^|?/??rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0`u?Hg@ @Qj/??X??AB????Yũ??,???rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	x	N} ???n?ݳ.??Ah?K6l??Ys?"?k??rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?X$??+øD???A.?v?????Y8?a?A
??rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????n3?????A?q75??Y?we???rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?.??????uu?b????AF????(??Y<Nё\???rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???bc^?????k?1??A???:U???Y?B??ˬ?rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?@I????{?p̲'??A????Xm??YE??S????rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???!o???,??f*D??AӇ.?o???YP??????rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????4c??_?sa???Ae??Q???Y?g@???rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0┹?F???l??+???A?{??c???Y??eO???rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0,g~5????B?Գ ??A???j??Y(-\Va3??rtrain 33*	????Ҙ?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatd?mlv$??!~?X? ?A@)(5
If??1?a$Np?>@:Preprocessing2E
Iterator::Rootj??????!??@q_GA@)L4H?S???1??O?-2@:Preprocessing2T
Iterator::Root::ParallelMapV2????߆??!?????`0@)????߆??1?????`0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???????!??_GP\P@)???Д???10h?2/?#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice??O?s'??!׿?9?E#@)??O?s'??1׿?9?E#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenatei???!??!B0?Ih0@)3??O??1[A??Ƶ@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???k???!44*F?@)???k???144*F?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????Tl??!?g?'I4@)9?d??)??1?1???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9.Cg׷?@I??DAZX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??Y?R???k!9?X??l??+???!0?GQgn??	!       "	!       *	!       2$	?????????y? ????{??c???!???K. @:	!       B	!       J$	??p%?[????vc??(-\Va3??!?ECƣT??R	!       Z$	??p%?[????vc??(-\Va3??!?ECƣT??b	!       JCPU_ONLYY.Cg׷?@b q??DAZX@Y      Y@qFG[?2V8@"?	
both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?24.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 