?&$	nĝ[????٨T????Y??L/1??!Xr?ߔ@$	??TDk@??ZP?}???????l@!c6i{?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0f0F$
m??	???W??A ??q????Y3܀?#??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Az?b??z?sѐ???A?}s???YD??]L3??rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?dU?????QhY????A\;Q???Y??kzPP??rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??BW"??????E??AkD0.???Y??x??rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~K?|???O??????AзKu???Y0c
?8???rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0 ?_>Yq??p?܁z??An???V??Yo??}U??rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C??À???Ӂ??V??A?E
e????Yx?????rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	ĔH??????????@??A?!????YDM??(#??rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
Y??L/1??.8??_???A?=?N????Y?HM??f??rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?&??|??V?y?U??AT?????Y??(??P??rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?_?|x?@o+?6??A>?ɋ???Y?&??n??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0o?j; @?B</???A9|҉S??Yk?ѯ???rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails07U?q7??t	????Ai?G5????Y?O??????rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??9]????
?????Am????U??Y7?C???rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0+/???]??{Cr2??A????#F??Y??`??rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??#?G???o?N\N??A??%?I??Y? :vP??rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Xr?ߔ@??j?j @A͓k
d???Y=HO?CĹ?rtrain 33*	B`?Т??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat3?-??!???7F?B@)<l"38??1??-?ty@@:Preprocessing2E
Iterator::Root. ?????!?}?
*?A@)?(B?v???1?;q%??2@:Preprocessing2T
Iterator::Root::ParallelMapV2?&S???!$?????0@)?&S???1$?????0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??$\???!A??jP@)Ҫ?t????1
N?(#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlicet{Ic????!?t4a"@)t{Ic????1?t4a"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??҇.???!??UŌF@)??҇.???1??UŌF@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::ConcatenateV??y???!A?[???+@)n??)"??1F}??Y?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?2?Pl??!??dq? 1@)ȳ˷>???1??Տ?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9];???K
@I%鈢-X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	-v???????zs??.8??_???!??j?j @	!       "	!       *	!       2$	9$?P29???!?? 9???=?N????!͓k
d???:	!       B	!       J$	??7??V????;
?????O??????!=HO?CĹ?R	!       Z$	??7??V????;
?????O??????!=HO?CĹ?b	!       JCPU_ONLYY];???K
@b q%鈢-X@Y      Y@qH??÷1@"?	
both?Your program is POTENTIALLY input-bound because 56.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?17.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 