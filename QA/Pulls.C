/*
 * This file is part of TPCCATracker package
 * Copyright (C) 2007-2020 FIAS Frankfurt Institute for Advanced Studies
 *               2007-2020 Goethe University of Frankfurt
 *               2007-2020 Ivan Kisel <I.Kisel@compeng.uni-frankfurt.de>
 *               2007-2014 Igor Kulakov
 *
 * TPCCATracker is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TPCCATracker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

void Pulls() {

  TStopwatch timer;
  timer.Start();

//  gROOT->LoadMacro("$VMCWORKDIR/gconfig/basiclibs.C");
//  basiclibs();

TStyle *plain = new TStyle("Plain","Plain Style(no colors/fill areas)");
//plain->SetCanvasBorderMode(0);
//plain->SetPadBorderMode(0);
plain->SetPadColor(0);
plain->SetCanvasColor(0);
plain->SetTitleColor(0);
plain->SetStatColor(0);
  // The parameter mode can be = ksiourmen  (default = 000001111)
    // k = 1;  kurtosis printed
    // k = 2;  kurtosis and kurtosis error printed
    // s = 1;  skewness printed
    // s = 2;  skewness and skewness error printed
    // i = 1;  integral of bins printed
    // o = 1;  number of overflows printed
    // u = 1;  number of underflows printed
    // r = 1;  rms printed
    // r = 2;  rms and rms error printed
    // m = 1;  mean value printed
    // m = 2;  mean and mean error values printed
    // e = 1;  number of entries printed
    // n = 1;  name of histogram is printed
plain->SetOptStat(111110);
plain->SetOptFit(0011);
plain->SetStatW(0.225);
plain->SetStatH(0.06);
//plain->SetStatX(0.7);
plain->SetOptTitle(0);
plain->cd();


  TString name = "HLTTPCCATrackerPerformance.root";

  TFile *f = new TFile(name.Data(),"read");

  FILE *ress, *pulls;
  ress = fopen("resolution.txt","w");
  pulls = fopen("pull.txt","w");
  
  const int NParameters = 5;
  const int NPerf = 2;
  
  string PerfName[NPerf];
  PerfName[0]="Sector Performance";
  PerfName[1]="Global Performance";

  string OutFileName[NPerf];
  OutFileName[0]="SectorTrackParamQuality.pdf";
  OutFileName[1]="GlobalTrackParamQuality.pdf";
  
  TH1D *hRes[NParameters][NPerf];
  TH1D *hPull[NParameters][NPerf];
  TF1  *fRes[NParameters][NPerf];
  TF1  *fPull[NParameters][NPerf];

  string sRes[NParameters][NPerf];
  string sPull[NParameters][NPerf];
  string sResFit[NParameters][NPerf];
  string sPullFit[NParameters][NPerf];
  
  string pullParNames[NParameters] = {"Y","Z","SinPhi","DzDs","QPt"};
  string resParNames[NParameters] = {"Y","Z","SinPhi","DzDs","Pt"};
  string ParVal[NParameters] = {" [cm]"," [cm]","",""," [c/Gev]"};

  string AxisNameRes[NParameters];
  string AxisNamePull[NParameters];

  AxisNameRes[0] = "Residual (y^{reco} - y^{mc}) [cm]";
  AxisNameRes[1] = "Residual (z^{reco} - z^{mc}) [cm]";
  AxisNameRes[2] = "Residual (sin(#phi)^{reco} - sin(#phi)^{mc}) ";
  AxisNameRes[3] = "Residual (dz/ds^{reco} - dz/ds^{mc})";
  AxisNameRes[4] = "Residual (p_{t}^{reco} - p_{t}^{mc})/p_{t}^{mc}";

  AxisNamePull[0] = "Pull y";
  AxisNamePull[1] = "Pull z";
  AxisNamePull[2] = "Pull sin(#phi)";
  AxisNamePull[3] = "Pull dz/ds";
  AxisNamePull[4] = "Pull q/p_{t}";

  string res = "res";
  string pull = "pull";

  for(int iPerf=0; iPerf<NPerf; iPerf++)
  {
    TString DirName = "HLTTPCCATrackerPerformance.root:/";
    DirName += PerfName[iPerf].data();
    DirName += "/TrackFit";
    TDirectory *dir = f->GetDirectory(DirName.Data());

    for(int i=0; i<NParameters; i++)
    {
      sRes[i][iPerf]=res+resParNames[i];
      sPull[i][iPerf]=pull+pullParNames[i];
      sResFit[i][iPerf]=res+resParNames[i]+PerfName[iPerf]+"GaussFit";
      sPullFit[i][iPerf]=pull+pullParNames[i]+PerfName[iPerf]+"GaussFit";
    }

    for(int i=0; i<NParameters; i++)
    {
      hRes[i][iPerf] = (TH1D*) dir->Get(sRes[i][iPerf].data());
      fRes[i][iPerf] = new TF1(sResFit[i][iPerf].data(),"gaus");
      fRes[i][iPerf]->SetLineColor(2);
      hRes[i][iPerf]->Fit(sResFit[i][iPerf].data(),"","",hRes[i][iPerf]->GetXaxis()->GetXmin(),hRes[i][iPerf]->GetXaxis()->GetXmax());
    
      hPull[i][iPerf] = (TH1D*) dir->Get(sPull[i][iPerf].data());
      fPull[i][iPerf] = new TF1(sPullFit[i][iPerf].data(),"gaus");
      fPull[i][iPerf]->SetLineColor(2);
      hPull[i][iPerf]->Fit(sPullFit[i][iPerf].data(),"","",hPull[i][iPerf]->GetXaxis()->GetXmin(),hPull[i][iPerf]->GetXaxis()->GetXmax());
    }

    fprintf(ress,"%s\n", PerfName[iPerf]);
    fprintf(pulls,"%s\n",PerfName[iPerf]);
    for(int i=0; i<NParameters; i++)
    {
      fprintf(ress,"%s	%.3g	%.3g\n", resParNames[i], fRes[i][iPerf]->GetParameter(1), fRes[i][iPerf]->GetParameter(2));
      fprintf(pulls,"%s	%.3g	%.3g\n", pullParNames[i], fPull[i][iPerf]->GetParameter(1), fPull[i][iPerf]->GetParameter(2));
    }
    fprintf(ress,"\n");
    fprintf(pulls,"\n");  
  }

  fclose(ress);
  fclose(pulls);

  TCanvas *c[NPerf];

  for(int iPerf=0; iPerf<NPerf; iPerf++)
  {
    c[iPerf] = new TCanvas(PerfName[iPerf].data(),PerfName[iPerf].data(),0,0,1200,800);
    c[iPerf] -> Divide(NParameters,2);
    c[iPerf] -> UseCurrentStyle();
    int NPad=0;

    for(int i=0; i<NParameters; i++)
    {
      NPad++;
      c[iPerf]->cd(NPad);
      hRes[i][iPerf]->GetXaxis()->SetTitle(AxisNameRes[i].data());
      hRes[i][iPerf]->GetXaxis()->SetTitleOffset(1);
      hRes[i][iPerf]->Draw();
      fRes[i][iPerf]->Draw("same");
    }
    for(int i=0; i<NParameters; i++)
    {
      NPad++;
      c[iPerf]->cd(NPad);
      hPull[i][iPerf]->GetXaxis()->SetTitle(AxisNamePull[i].data());
      hPull[i][iPerf]->GetXaxis()->SetTitleOffset(1);
      hPull[i][iPerf]->Draw();
      fPull[i][iPerf]->Draw("same");
    }

    c[iPerf]->SaveAs(OutFileName[iPerf].data());
  }
    
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();
  printf("RealTime=%f seconds, CpuTime=%f seconds\n",rtime,ctime);
}
