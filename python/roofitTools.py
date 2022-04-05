import ROOT
import os

#import numpy as np
#
#def chiSquare(hobs,href,hsysts):
#    
#    nbins=hobs.GetNbinsX()
#    statCov=np.zeros(nbins,nbins)
#    systCov=[ np.zeros(nbins,nbins) for is in range(len(hsysts)) ]
#    diff=np.zeros(nbins)
#
#    for xbin in range(nbins):
#        
#        statCov[xbin][xbin]+=(hobs.GetBinError(xbin+1))**2
#
#        for ybin in range(nbins):
#
#            if ybin==xbin:
#                statCov[xbin][xbin] += (href.GetBinError(xbin+1))**2
#                diff[xbin]=hobs.GetBinContent(xbin+1)-href.GetBinContent(xbin+1)
                
        
                


def showFitResult(w,varDesc,data,pdfCompList,extraText,outName,xran=None):

    """ 
    Displays the results of the PDF fits for a list of PDF components
    varDesc=(name,title) of the x-axis
    data - a RooDataset
    pdfCompList=[ (pdf name, pdf components, color, linestyle), ... ]
    extraText - print this to canvas
    xran - range (min,max)
    outName - output file name
    """

    vname,vtit=varDesc

    #plot slices one by one to compare with the model
    c = ROOT.TCanvas('c','c',500,500)
    p1 = ROOT.TPad('p1','p1',0.0,0.85,1.0,0.0)
    p1.Draw()
    c.cd()
    p2 = ROOT.TPad('p2','p2',0.0,0.85,1.0,1.0)
    p2.Draw()

    p1.cd()
    p1.Clear()
    p1.SetRightMargin(0.04)
    p1.SetLeftMargin(0.12)
    p1.SetTopMargin(0.008)
    p1.SetBottomMargin(0.1)
    p1.SetGridx(True)
    frame=w.var(vname).frame()
    data.plotOn(frame)
    for i in range(len(pdfCompList)):
        name,comp,ci,ls=pdfCompList[i]
        if not ci: ci=ROOT.kBlue
        if not ls: ls=1
        if comp:
            w.pdf(name).plotOn(frame,ROOT.RooFit.LineColor(ci),ROOT.RooFit.LineStyle(ls),ROOT.RooFit.Components(comp))
        else:
            w.pdf(name).plotOn(frame,ROOT.RooFit.LineColor(ci),ROOT.RooFit.LineStyle(ls))
    frame.Draw()
    frame.GetXaxis().SetTitle(vtit)
    frame.GetYaxis().SetTitle("Entries")
    frame.GetYaxis().SetTitleOffset(1.0)
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.GetXaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetLabelSize(0.04)
    if xran : frame.GetXaxis().SetRangeUser(xran[0],xran[1])

    tex=ROOT.TLatex()
    tex.SetTextFont(42)
    tex.SetTextSize(0.04)
    tex.SetNDC()
    tex.SetTextAlign(ROOT.kHAlignRight+ROOT.kVAlignCenter)
    tex.DrawLatex(0.95,0.94,'#bf{CMS} #it{simulation preliminary}')
    tex.DrawLatex(0.95,0.90,'#chi^{2}=%3.2f'%frame.chiSquare())
    if len(extraText)>0 : tex.DrawLatex(0.95,0.86,extraText)


    p2.cd()
    p2.Clear()
    p2.SetBottomMargin(0.005)
    p2.SetRightMargin(0.04)
    p2.SetLeftMargin(0.12)
    p2.SetTopMargin(0.05)
    p2.SetGridx(True)
    p2.SetGridy(True)
    
    hpull = frame.pullHist()
    pullFrame = w.var(vname).frame()
    pullFrame.addPlotable(hpull,"P") ;
    pullFrame.Draw()
    pullFrame.GetYaxis().SetTitle("Pull")
    pullFrame.GetYaxis().SetTitleSize(0.3)
    pullFrame.GetYaxis().SetLabelSize(0.25)
    pullFrame.GetXaxis().SetTitleSize(0)
    pullFrame.GetXaxis().SetLabelSize(0)
    pullFrame.GetYaxis().SetTitleOffset(0.15)
    pullFrame.GetYaxis().SetNdivisions(4)
    pullFrame.GetYaxis().SetRangeUser(-3.1,3.1)
    pullFrame.GetXaxis().SetTitleOffset(0.8)
    if xran : pullFrame.GetXaxis().SetRangeUser(xran[0],xran[1])

    c.Modified()
    c.Update()
    for ext in ['png', 'pdf']:
        c.SaveAs("%s.%s"%(outName,ext))
    c.Delete()


def shushRooFit():

    """stop all the messaging from RooFit"""

    ROOT.RooMsgService.instance().setSilentMode(True)
    ROOT.RooMsgService.instance().getStream(0).removeTopic(ROOT.RooFit.Minimization)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Minimization)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.ObjectHandling)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.DataHandling)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Fitting)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Plotting)
    ROOT.RooMsgService.instance().getStream(0).removeTopic(ROOT.RooFit.InputArguments)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.InputArguments)
    ROOT.RooMsgService.instance().getStream(0).removeTopic(ROOT.RooFit.Eval)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Eval)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Integration)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.NumIntegration)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Caching)
