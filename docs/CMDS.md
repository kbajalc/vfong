python3 setup.py build_ext --inplace -f >/tmp/vf_build.log 2>&1 && python3 feature_extraction.py -o features/features_s8.dat -s 8
zsh test_classifiers.sh



python3 setup.py build_ext --inplace -f >/tmp/vf_build.log 2>&1 && python3 feature_extraction.py -o features/features_s8.dat -s 8