?*$	F)n)4?@@?`?C???*?"b??!i?^`Vx@$	?+:?e?@66??????iP?????!?P??/#@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?4*p??	@???&?&??A-`?????Y?ɋL????rtrain 53"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0i?^`Vx@???O?n@AԵ?>U?@Y:?S?????rtrain 54"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0^??_????fԼ??A?d#٣??Y&??)??rtrain 55"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/(__?@?=?$@M??Af???i??Y???:q9??rtrain 0"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/i?'??~
@??T????Ar2q? ???Y{L?4???rtrain 1"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?@?t????d?,?i??A? ????Y?׹i3N??rtrain 2"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?	?s3???t?_?J??A:???u??Yh????[??rtrain 3"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/	*?"b???N[#?q??A!˂?????Yt%?????rtrain 4"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/
??0a4+???KK??A?{??˙??Y?W\???rtrain 5"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/!?> ?	@?3???l??A?R??????Y?4}v???rtrain 6"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/???R??@?S??Y???A?l??Ԃ??Y?x#????rtrain 7"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/3?}ƅ?@31]??_@A?5?????Y!?> ?M??rtrain 8"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?????@?S??y??A)?'?$???Y??R?1???rtrain 9"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???)??@?q?d????A??8?#??Y???W:??rtrain 10"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0&???????:U?g???A9??? ??Y??3?????rtrain 11"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0;?p?G @??sC??A\:?<c??Ymp"?????rtrain 12"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ߨ??5??Wzm6Vb??Ag'??????Y?ZӼ???rtrain 13*Zd;_+?@)      `=2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?qS??!???H??@@)???(yu??1???k??7@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(F?̱<??!?N?s9@)?,^,??1<b?;3?0@:Preprocessing2E
Iterator::Root]?`7l???!?sE8s8@)5{????1/??g??+@:Preprocessing2T
Iterator::Root::ParallelMapV2?!Y???!0?Ec$@)?!Y???10?Ec$@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?ܚt["??!CSIZ?"@)?ܚt["??1CSIZ?"@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map ?vi???!?	?#?#$@),g~5??1ʵ??!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicenk?K??!???? @)nk?K??1???? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip}%??? @!ߡw?ftP@)?^?iN^??1?vD??t@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeath?o}Xo??!#?Y???)?M?#~Ţ?1?.WnF5??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicet&m????!@j1?d??)t&m????1@j1?d??:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchTƿϸ??![?tW8??)Tƿϸ??1[?tW8??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range|?ԗ??j?!}笛)ѹ?)|?ԗ??j?1}笛)ѹ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9D?%c?G@I???$?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	'~??k	???T?m\????N[#?q??!???O?n@	!       "	!       *	!       2$	?jQ????y?D????!˂?????!Ե?>U?@:	!       B	!       J$	??M~4E???̈́??????ɋL????!:?S?????R	!       Z$	??M~4E???̈́??????ɋL????!:?S?????b	!       JCPU_ONLYYD?%c?G@b q???$?X@Y      Y@q????i@@"?	
both?Your program is POTENTIALLY input-bound because 52.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?32.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 