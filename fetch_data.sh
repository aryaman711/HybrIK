cd /content/HybrIK

# username and password input as arguments
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

username=$(urle "$1")
password=$(urle "$2")

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' -O '/content/HybrIK/mpips_smplify_public_v2.zip' --no-check-certificate --continue
unzip mpips_smplify_public_v2.zip -d smplx_files
mv smplx_files/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ./model_files
rm -rf *.zip mpips_smplify_public_v2
