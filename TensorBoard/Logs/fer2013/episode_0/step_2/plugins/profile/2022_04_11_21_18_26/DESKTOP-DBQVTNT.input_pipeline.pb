$	?>+n$@?d?T?????h??#@!?@?]?U%@$	??*???2?`]??k??3?c??!?/喑???"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?RD?UD$@????A2?????"@Y+MJA????r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?v??-?$@?'?H0???A?_"?:w#@Y??D?Ɵ??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1@?P?5$@%?s}???A<jL???"@Y?I?????r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?F?q?%@????????As߉Y?#@Y?º??Ȩ?r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?N?6H$@????5???A??z??"@Y??j?	???r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1<?R?!>$@???4}??A?
*?~?"@Y\?d8?Ϩ?r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???h??#@9|҉???A?i3NC?"@Y??l??T??r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	u??.??#@w?$???A?5??Ң"@Y?]=?1??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
?@?]?U%@5z5@i???A?qR???#@YhwH1@??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1+??X?r$@????(???Aog_y?#@YAc&Q/??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?O=? $@???P???AF$
-??"@Y???5?e??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1=???$@???5???A?t???d#@Y??O=Ҩ?r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1kGq?:?$@??^??A????;#@Y????5w??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1Q?Hm$@???????A??5"#@Y??_?????r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1{L?4?$@Zh?44??A!??=@7#@Y?\S ????r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1:w?^?$@??&?????Au?׃I?"@Y?wak???r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1OZ??"$@?T?=???A?vj.7?"@Y???x??r	train 517*	n??J??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat,??26t??!???7n?D@)}???????1?c
rMB@:Preprocessing2T
Iterator::Root::ParallelMapV2닄??K??!??|n?/@)닄??K??1??|n?/@:Preprocessing2E
Iterator::Root?z?????!????~h?@)??o?4(??1??^Վ?.@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?P1?߄??!qGl?0)@)?P1?߄??1qGl?0)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipA*Ŏ???!?O?U?%Q@)?)1	??1????/?$@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*t]???Ե?!??+Q@)t]???Ե?1??+Q@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap3??J&??!^?d?0@)0?^|???12?t???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9SQ?????I?Qv??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	f?]6??>UM???????4}??!%?s}???	!       "	!       *	!       2$	??I`#@?????????i3NC?"@!?qR???#@:	!       B	!       J$	@TvVg??b?W?G?h???l??T??!+MJA????R	!       Z$	@TvVg??b?W?G?h???l??T??!+MJA????b	!       JCPU_ONLYYSQ?????b q?Qv??X@