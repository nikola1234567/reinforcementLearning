?*$	?>2j\@"⼃???Hū????!J?y?@$	??f??Q@?g??????rR'CC??!?5?~?	@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ަ??q@??????APoF?W	??Y???]????rtrain 53"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0mV}???@	]????AT?J?ó??Y?gz??L??rtrain 54"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0J?y?@ԝ'??? @AA?>???Y??x??[??rtrain 55"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/UL??p?@??=????AR*?	????Y????4c??rtrain 0"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/7p??'@&?lscZ@A??R?1???Y??"????rtrain 1"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/_z?s?@???????A<?y8A??Y'K?????rtrain 2"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/-???as@???t?4??A??j??Y?ǘ?????rtrain 3"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/	?5>???????Dr??Ar4GV~??Y@OI???rtrain 4"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/
?Y.??????׹i3??A???$????Y>\r?)??rtrain 5"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/??a????>+N?V??AC?+j??Y?fe?????rtrain 6"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/Hū?????N^t??A??E_A??Y?cϞˬ?rtrain 7"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?e?c]\ @?M(D?a??A6u???Y?jׄ?ư?rtrain 8"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/F?7?k @{?/L?
??Al???C6??Y?M???P??rtrain 9"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?J?4Q???l???AY??9?}??Y??M~?N??rtrain 10"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~1[? @?ƠB??A?n?UfJ??YM???$??rtrain 11"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Ft?@??6????A=??? ???Y?X S??rtrain 12"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0a?d7?@5?l?/???APoF????Y?/?'??rtrain 13*	?|?5???@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap=??MZ??!?n????@@)??t????1o@A{?;@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatH?)s????!d?b??4@)P?mp???1V?hN%?1@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapk?ѯ-??!ňg?#6@)g)YNB??1ʡ?~U?1@:Preprocessing2E
Iterator::RootHk:!t??!??V???1@)$G:#/??1kPj?m"@:Preprocessing2T
Iterator::Root::ParallelMapV2m??p???!??B/!@)m??p???1??B/!@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??luy @!????N@)??X???1???׳=@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceܻ}????!t?Nv@)ܻ}????1t?Nv@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?q?߅???!??C?8?@)??z??&??1??7?|@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*ß?????!s???p?@)ß?????1s???p?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice7l[?? ??!?;??v??)7l[?? ??1?;??v??:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch????W??!???eƪ??)????W??1???eƪ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeUܸ???p?!?>??ʾ?)Uܸ???p?1?>??ʾ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Kx@Idq??=\X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?6??h??rϰ??????N^t??!&?lscZ@	!       "	!       *	!       2$	&??i[E??y?X?????r4GV~??!A?>???:	!       B	!       J$	?娙?????A`#ĉ?>\r?)??!??"????R	!       Z$	?娙?????A`#ĉ?>\r?)??!??"????b	!       JCPU_ONLYY??Kx@b qdq??=\X@Y      Y@qb?R???@"?	
both?Your program is POTENTIALLY input-bound because 57.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?32.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 