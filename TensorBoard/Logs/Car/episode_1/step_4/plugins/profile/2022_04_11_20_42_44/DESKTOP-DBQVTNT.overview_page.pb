?&$	??:????(h?p???2?m?????!??߆O@$	?r?yˊ@?6R????:"???[@!?s^?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???W:????	?????AS???"???Y+l? [??rtrain 17"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??z?p?@?????}??A[A????Y?W?}W??rtrain 18"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?P????@??|~???A???3???Y9??!???rtrain 19"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Ĭ@?(?????A#/kb????Y?<$}??rtrain 20"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??߆O@c ?=???A??N?`???Yٖg)Y??rtrain 21"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?]???'i??????A?_=?[???Y????Y.??rtrain 22"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0B`??"?????8ӄm??A?rg&N??Y????i??rtrain 23"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??$y?????g????A??H¾??Y???G?C??rtrain 24"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???8?*???B˺??A??
~b??Y?SV??D??rtrain 25"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0<???????0~????A?]?9????Y??n?!??rtrain 26"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??'*??]?????A?~?????Y?Z?}??rtrain 27"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0#?J %v??Mf???Z??A?@ C???Y???,'???rtrain 28"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?s???)??d?w<??A???V`H??Y8fٓ????rtrain 29"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02?m??????)????AN??oD???Y@???൫?rtrain 30"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??rg&??~(F????A?lt?Oq??Yj.7갪?rtrain 31"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0TUh ?M??-B?4-??A??\????Y??4???rtrain 32"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???X??@??g%?x??Aʉv???Y?}?
Ĳ?rtrain 33*	?&1?U?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??Z(???!M?/???C@)?eo)g??1Z?7$??@:Preprocessing2E
Iterator::Root?0&????!ʬ.f?J?@)V???4???1QqX?0@:Preprocessing2T
Iterator::Root::ParallelMapV2???
G???!?v??J|.@)???
G???1?v??J|.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip/ܹ0???!?Tt?R-Q@)eS??.??1?D??#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSliceePmp"???!EQ??v?"@)ePmp"???1EQ??v?"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?(??????!~t?
1 @)?(??????1~t?
1 @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[4]::ConcatenateZ??mē??!}??S0@)O$?jf-??1h?_?_?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????e??!??O3@)7R?Hڍ??1?2?P?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??"???
@Ix??9?(X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	$(,?????S??yW????)????!?(?????	!       "	!       *	!       2$	V|??K????pL?r????lt?Oq??!??N?`???:	!       B	!       J$	"9G˯???G?F???j.7갪?!?<$}??R	!       Z$	"9G˯???G?F???j.7갪?!?<$}??b	!       JCPU_ONLYY??"???
@b qx??9?(X@Y      Y@q??0QN?5@"?	
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
Refer to the TF2 Profiler FAQb?21.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 