?*$	?}y]?@???F????I??????!?&ݖ??@$	???H?,
@?u???o???0?S????!?S;??D"@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~l??@p
+T??A}@?3i???YN??;P??rtrain 53"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0s?	M?@???+H??AwN?@??@Y#2??????rtrain 54"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???4????7ӅX??A)??聏??Y???*???rtrain 55"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/f?y???@=~oӟ]@AI?L??&??Y!yv???rtrain 0"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/2s??c???
????=??A???5???Y?M???P??rtrain 1"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/=$}Z?????5??A?0??????Y!v??y???rtrain 2"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/oc?#?w??9?]????A??????Y????A_??rtrain 3"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/	?n???@w??-uP??AH?Sȕz??Y[?{c ??rtrain 4"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/
?&ݖ??@?uS?ke??A?7k??@YZ?!?[=??rtrain 5"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/̵h?V@?%?"?d??A?b?0???YҎ~7???rtrain 6"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/kծ	i?@??)??z??AyX?5?{??YްmQf???rtrain 7"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/???@?-@??o?????A??OV???Y? -??rtrain 8"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?	1?T?@?<??- ??A????????YK??z2???rtrain 9"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Ƽ?8$????z?????A????K??Y1?:9Cq??rtrain 10"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?I???????f?ܶ??AZd;?O??Y??|?R???rtrain 11"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0*?t??????S???>??A?F??R^??Y?Z?a/??rtrain 12"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails04J??e???@?"??A??0?q???Y? ??q4??rtrain 13*	?ʡE???@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?|$%=???!????
9@)$??:??1??"?˞5@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??.?D??!V$/o??4@)??.?D??1V$/o??4@:Preprocessing2E
Iterator::Rootݵ?|г??!?̛??!<@)y?&1,??1?h?`??0@:Preprocessing2T
Iterator::Root::ParallelMapV2Ș?????!???#?&@)Ș?????1???#?&@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap~?.r??!??³-{?@)???!9???1?R?l<0$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipY?e0F @!l?h|?O@) s-Z????1?d?T@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???s?v??!??%?# @)??cx?g??1??????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???P???! ? .?Y@)???P???1 ? .?Y@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatΈ?????!3О\ͪ@)M??ӀA??1??=?t?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??\5ρ?!???ot??)??\5ρ?1???ot??:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???????!T??g??)???????1T??g??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/PR`Li?!"'L˸?)/PR`Li?1"'L˸?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??iZy?
@I??,5?*X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	h&j?[+??7"???????5??!=~oӟ]@	!       "	!       *	!       2$	??ZM???????IF??Zd;?O??!?7k??@:	!       B	!       J$	H??"???WQ-??Q??1?:9Cq??![?{c ??R	!       Z$	H??"???WQ-??Q??1?:9Cq??![?{c ??b	!       JCPU_ONLYY??iZy?
@b q??,5?*X@Y      Y@q?x??l8=@"?	
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?29.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 