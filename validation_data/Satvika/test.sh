for file in *.wav;do
    outfile="/home/adil/Downloads/validation_data/spec/7u/${file%.*}.png"
    title_in_pic="${file%.*}"
    sox "$file" -n spectrogram -t "$title_in_pic" -o "$outfile" -x 2000
done
