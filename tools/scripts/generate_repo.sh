#!/bin/bash

# this script creates a repository from the latest built
# i.e. it collects all .jars into one folder 

tools_dir=$PWD
base_dir=$(dirname $(dirname $PWD/..)) 
repo_name=$(basename $base_dir)
folder="generated/workspacerepo"
local_repo=$base_dir/cnf/localrepo

echo "Generating repo $repo_name for workspace $base_dir"

mkdir -p $folder/$repo_name


# copy all built .jars
echo "Copying latest builts"
for dir in `ls $base_dir`
do
	if [[ -d $base_dir/$dir/generated ]]; then
		if [[ ($dir != "tools") && ($dir != "test") ]]; then
			cd $base_dir/$dir/generated/
			for jar in `ls *.jar 2> /dev/null`
			do
				echo "Copy $jar..."
				dirname=$(basename $jar .jar)
				mkdir -p $tools_dir/$folder/$dirname
				cp $base_dir/$dir/generated/$jar $tools_dir/$folder/$dirname/$dirname-latest.jar
			done
		fi
	fi
done

# generate index
cd $tools_dir/$folder/
java -jar $tools_dir/scripts/org.osgi.impl.bundle.repoindex.cli-0.0.4.jar **/*.jar --pretty -n $repo_name