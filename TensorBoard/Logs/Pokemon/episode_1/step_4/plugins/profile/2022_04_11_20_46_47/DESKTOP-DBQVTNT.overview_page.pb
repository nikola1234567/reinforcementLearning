?*$	??R??D@???U??G?J??q??!Ԝ??|@$	??\?r
@tᮩf?????3FM??!cD?L?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0M?????
@kIG9?? @A?c??e??Ywj.7???rtrain 53"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0l?????@H?Ȱ????AUMu???YR?>?G???rtrain 54"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????́@??;P????A? U??YC?O?}:??rtrain 55"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/i;???n???w?????A?d??7i??Y?&2s?˫?rtrain 0"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/p?x?0? @??,
????ADܜJ???Y7??nf???rtrain 1"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/CV?zN?@??2?6? @A?? ?S???Y??"?ng??rtrain 2"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/tD?K???????k????A-???YHN&nİ?rtrain 3"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/	G?J??q??ۉ??H???A;ŪA??Y??x?@e??rtrain 4"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/
Ҩ?ɶ???\?????A?z?f?l??Y/??ا?rtrain 5"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/Ԝ??|@ 'L͊ @A?ͪ?? @YJ'L5???rtrain 6"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/Uh ??@?|?q? @A46<???Y?;FzQ??rtrain 7"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/l????@?F? ???A??????Y????W???rtrain 8"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?eM,?5@V??????A:????YGW??:??rtrain 9"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Ŧ?B  @9{g?U???AcG?P???Y-????)??rtrain 10"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?GnM?????C?Ö??AcB?%U???Y$?@???rtrain 11"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?%?` @?D?<??Ae6?$???YЀz3j???rtrain 12"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Ŏơ~w @??켍???A?Pj/?m??Y?n??S??rtrain 13*	???S#ϧ@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???8a???!E??mp}=@)P??|zl??1[?T??:@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???^a???!?i?go;@)???V???1Ocv?-5@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map!??????!;#???9-@)??R%????1?L???*@:Preprocessing2E
Iterator::Root?D????!?{-,d16@)y ?H??1?h??_?'@:Preprocessing2T
Iterator::Root::ParallelMapV2????5??!Ў??h?$@)????5??1Ў??h?$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???fd???!Q9??ИO@)횐?t??1j?tt&@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceq:?V?S??!?R??@)q:?V?S??1?R??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???5???!H?	4?\@)???5???1H?	4?\@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat	R)v4??!???(???)?T2 Tq??1??????:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??>V?ۀ?!_!^??I??)??>V?ۀ?1_!^??I??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice`!sePm??! J?)???)`!sePm??1 J?)???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range ?o_?i?!??	?u??) ?o_?i?1??	?u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Vr&3@I{Im?fX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	:???)???'g!???ۉ??H???!??2?6? @	!       "	!       *	!       2$	S<{{$????˱+?U???d??7i??!?ͪ?? @:	!       B	!       J$	ث=??ܶ???ݻ???/??ا?!J'L5???R	!       Z$	ث=??ܶ???ݻ???/??ا?!J'L5???b	!       JCPU_ONLYY??Vr&3@b q{Im?fX@Y      Y@q&4?uc>@"?	
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?30.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 