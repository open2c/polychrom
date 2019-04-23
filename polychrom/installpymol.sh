#!/bin/bash -e
# This script was downloaded from http://www.pymolwiki.org/index.php/Linux_Install
pymoldir=$HOME/programming/pymol
modules=$pymoldir/modules
svnpymol=svnpymol
svnfreemol=svnfreemol
pymolscriptrepo=Pymol-script-repo 

###################################################
[ -d $pymoldir ] || mkdir -p $pymoldir
[ -d $HOME/bin ] || mkdir $HOME/bin

###### Install required system packages
sudo apt-get install subversion build-essential python-dev python-pmw libglew-dev freeglut3-dev libpng-dev libfreetype6-dev

###### Checkout pymol svn
svn co https://svn.code.sf.net/p/pymol/code/trunk/pymol $pymoldir/$svnpymol
###### Build and install pymol
cd $pymoldir/$svnpymol
python setup.py build install --home=$pymoldir --install-lib=$modules
export PYTHONPATH=$modules:$PYTHONPATH
python setup2.py install
install pymol $pymoldir/

########## Setup freemol - for MPEG support ############
svn co svn://bioinformatics.org/svnroot/freemol/trunk $pymoldir/$svnfreemol
cd $pymoldir/$svnfreemol/src/mpeg_encode
export FREEMOL=$pymoldir/$svnfreemol/freemol
./configure
make
make install

########## Install Pymol-script-repo ############
git clone git://github.com/Pymol-Scripts/Pymol-script-repo.git $pymoldir/$pymolscriptrepo

## Make a shortcut to an extended pymol execution
echo "#!/bin/bash" >> $pymoldir/pymolMPEG.sh
echo "if [ ! -f $HOME/.local/share/applications/pymolsvn.desktop ];" >> $pymoldir/pymolMPEG.sh
echo "then" >> $pymoldir/pymolMPEG.sh
echo "ln -s $pymoldir/pymolsvn.desktop $HOME/.local/share/applications/pymolsvn.desktop" >> $pymoldir/pymolMPEG.sh
echo "fi" >> $pymoldir/pymolMPEG.sh
echo "export FREEMOL=$pymoldir/$svnfreemol/freemol" >> $pymoldir/pymolMPEG.sh
echo "export PYMOL_GIT_MOD=$pymoldir/$pymolscriptrepo/modules" >> $pymoldir/pymolMPEG.sh
echo '#export PYTHONPATH=$PYTHONPATH:/path/to/shared/python/site-packages/PIL' >> $pymoldir/pymolMPEG.sh
echo '#export PYTHONPATH=$PYTHONPATH:/path/to/shared/python/site-packages/lib-dynload' >> $pymoldir/pymolMPEG.sh
echo "export PYTHONPATH=$pymoldir/$pymolscriptrepo/modules"':$PYTHONPATH' >> $pymoldir/pymolMPEG.sh
echo "export PYTHONPATH=$pymoldir/$pymolscriptrepo"':$PYTHONPATH' >> $pymoldir/pymolMPEG.sh
echo '#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/shared/lib/pymollib' >> $pymoldir/pymolMPEG.sh
echo '#export LIBGL_ALWAYS_INDIRECT=no' >> $pymoldir/pymolMPEG.sh
tail -n +2 $pymoldir/pymol >> $pymoldir/pymolMPEG.sh
chmod u+x $pymoldir/pymolMPEG.sh 

## Make a link, so we execute pymol with the freemol env exported
ln -s $pymoldir/pymolMPEG.sh $HOME/bin/pymol

## Make a pymolsvn.desktop
echo "#!/bin/env xdg-open" >> $pymoldir/pymolsvn.desktop
echo "[Desktop Entry]" >> $pymoldir/pymolsvn.desktop
echo "Version=1.5.x" >> $pymoldir/pymolsvn.desktop
echo "Name=PyMOL Molecular Graphics System - Open source" >> $pymoldir/pymolsvn.desktop
echo "GenericName=Molecular Modeller" >> $pymoldir/pymolsvn.desktop
echo "Comment=Model molecular structures and produce high-quality images of them" >> $pymoldir/pymolsvn.desktop
echo "Type=Application" >> $pymoldir/pymolsvn.desktop
echo "Exec=env $pymoldir/pymolMPEG.sh" >> $pymoldir/pymolsvn.desktop
echo "Icon=$pymoldir/$pymolscriptrepo/files_for_examples/pymol.xpm" >> $pymoldir/pymolsvn.desktop
echo "MimeType=chemical/x-pdb" >> $pymoldir/pymolsvn.desktop
echo "Categories=Education;Science;Chemistry;" >> $pymoldir/pymolsvn.desktop
echo "Terminal=false" >> $pymoldir/pymolsvn.desktop

## Make a startup files, which is always executed on startup.
t="'"
echo "import sys,os" >> $modules/pymol/pymol_path/run_on_startup.py
echo "import pymol.plugins" >> $modules/pymol/pymol_path/run_on_startup.py
echo "pymol.plugins.preferences = {'instantsave': False, 'verbose': False}" >> $modules/pymol/pymol_path/run_on_startup.py
echo "pymol.plugins.autoload = {'apbs_tools': False}" >> $modules/pymol/pymol_path/run_on_startup.py
echo "pymol.plugins.set_startup_path( [$t$pymoldir/$pymolscriptrepo/plugins$t,$t$modules/pmg_tk/startup$t] )" >> $modules/pymol/pymol_path/run_on_startup.py
echo "pymol.plugins.preferences = {'instantsave': True, 'verbose': False}" >> $modules/pymol/pymol_path/run_on_startup.py
