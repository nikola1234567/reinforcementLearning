$	???>y$@~G!Ҳ????v???#@!U????%@$	???0???,?I~??%??	LP??!gE?????"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1k*??.j$@E?u?????A?~k'J#@Y??O8????r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1! _B?$@4??X????A?(?1C#@Yyͫ:???r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1'?????#@????R??A??0?"@Y[닄????r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails18?Jw??#@ۤ?????A
+TT?"@Y?|A	??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???T?(%@?w?7N???AB??#@Yv28J^???r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1U????%@\??.????A??)t^#$@Y???;??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?J?4?$@?$??????A???J#@YZ?!?[=??r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	?B?=?$@?B˺???A??ht?"@Y?Ēr?9??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
#?dTv$@qTn?????Au["??"@Y?f????r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1t{Ic4$@?@+0d???A???e??"@Y=dʇ?j??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1b.?I$@!撪?&??A???s??"@Y(?x?ߢ??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?u7Om$@??o?????Ao+?6#@Y?]L3????r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????P$@?խ??^??A???FX?"@Y???/fK??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1B?/h!?$@?ꭁ????A(ђ??#@Y.q???"??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?v???#@c*???[??A????5"@Y? ?????r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1'???K?#@)B?v????A??RAE}"@Y]P?2???r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1{j??Uq$@F_A??h??A?9[@h#@Y???{b??r	train 517*	?|?5^?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatv?????!R?`xD@)M?*??.??1Ԣ????A@:Preprocessing2E
Iterator::Root?ՏM?#??!?O?X?B@)[?}s??1ݗqI??2@:Preprocessing2T
Iterator::Root::ParallelMapV2<??f???!T?-??^2@)<??f???1T?-??^2@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??#???!?]a$@)??#???1?]a$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipj??%??!?c??SO@)Uj?@+??1v?7???!@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*K??^b,??!?#+Q??@)K??^b,??1?#+Q??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapmp"?????!߂Q G+@)?????~??1̤??~?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?w?v???I???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??t.????g??|???@+0d???!?B˺???	!       "	!       *	!       2$	'KV???"@D???q-??????5"@!??)t^#$@:	!       B	!       J$	??+V???fz?y?b?? ?????!Z?!?[=??R	!       Z$	??+V???fz?y?b?? ?????!Z?!?[=??b	!       JCPU_ONLYY?w?v???b q???X@