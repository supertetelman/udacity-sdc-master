for test in `ls nn*py`; do
    python ${test}
    if [ "${?}" != "0" ]; then
        echo "Something is wrong in ${?}."
	exit
    fi
done
