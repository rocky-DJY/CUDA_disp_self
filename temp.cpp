//
// Created by maxwell on 11/4/20.
//
int main(){
    for (int i = this->offsety; i < left.rows-offsety; i++) { // row index
        // first get the one row image data
        for(int T=0;T<left.cols;T++){
            L_0[T]= (float)left_RGB[0].at<u_char>(i,T)/uni_size;
            R_0[T]= (float)right_RGB[0].at<u_char>(i,T)/uni_size;
            L_1[T]= (float)left_RGB[1].at<u_char>(i,T)/uni_size;
            R_1[T]= (float)right_RGB[1].at<u_char>(i,T)/uni_size;
            L_2[T]= (float)left_RGB[2].at<u_char>(i,T)/uni_size;
            R_2[T]= (float)right_RGB[2].at<u_char>(i,T)/uni_size;
        }
        image_y=i-offsety;
        //plot(L,"Ldata",cv::Size(1920,256),4);
        //plot(R,"Rdata",cv::Size(1920,256),4);
        for (int j = offsetx; j < left.cols-offsetx; j++) {  // col index
            image_x=j-offsetx;
            image_left_roi.at<u_char>(image_y,image_x)=image_left_gray.at<u_char>(i,j);
            //  get the left block data
            for(int front_index=j,back_index=j,front=(int)len/2,back=(int)len/2;back_index>=j-len/2;front_index++,back_index--){
                if(front_index>left.cols-1){
                    temp_L_0[front]=0.0000f;
                    temp_L_1[front]=0.0000f;
                    temp_L_2[front]=0.0000f;
                    front++;
                }
                else {
                    temp_L_0[front]=L_0[front_index];
                    temp_L_1[front]=L_1[front_index];
                    temp_L_2[front]=L_2[front_index];
                    front++;
                }
                if(back_index<0){
                    temp_L_0[back]=0.0000f;
                    temp_L_1[back]=0.0000f;
                    temp_L_2[back]=0.0000f;
                    back--;
                }
                else {
                    temp_L_0[back] = L_0[back_index];
                    temp_L_1[back] = L_1[back_index];
                    temp_L_2[back] = L_2[back_index];
                    back--;
                }
            }
            // plot(temp_L,"L_temp",cv::Size(len,256),1);
            for (int m = MIN_DISPARITY; m <=MAX_DISPARITY; m++) {
                int current_right = j-m;
                if(current_right>=offsetx){
                    // get the right block data
                    for(int front_index=current_right,back_index=current_right,front=(int)len/2,back=(int)len/2;
                        back_index>=current_right-len/2;front_index++,back_index--){
                        if(front_index>left.cols-1){
                            temp_R_0[front]=0.0000f;
                            temp_R_1[front]=0.0000f;
                            temp_R_2[front]=0.0000f;
                            front++;
                        }
                        else {
                            temp_R_0[front]=R_0[front_index];
                            temp_R_1[front]=R_1[front_index];
                            temp_R_2[front]=R_2[front_index];
                            front++;
                        }
                        if(back_index<0){
                            temp_R_0[back]=0.0000f;
                            temp_R_1[back]=0.0000f;
                            temp_R_2[back]=0.0000f;
                            back--;
                        }
                        else{
                            temp_R_0[back]=R_0[back_index];
                            temp_R_1[back]=R_1[back_index];
                            temp_R_2[back]=R_2[back_index];
                            back--;
                        }
                        // cout<<"front: "<<front<<" back: "<<back<<"   len: "<<len<<endl;
                    }
                    // plot(temp_R,"R_temp",cv::Size(len,256),1);
                    // corre(temp_L,temp_R,res,lags,len);
                    // int curr_trans_val=trans_val(res,lags,len);
                    // the disstence of the  two arrays trans val and assuming trans;
                    float diss0=abs(calculate_corss_correlation(temp_L_0,temp_R_0,len));  // 数值越大表示代价越小
                    float diss1=abs(calculate_corss_correlation(temp_L_1,temp_R_1,len));
                    float diss2=abs(calculate_corss_correlation(temp_L_2,temp_R_2,len));
                    float aver_corr=(diss0+diss1+diss2)/3;
                    // float sum_abs_co=abs(diss0-aver_corr)+abs(diss1-aver_corr)+abs(diss2-aver_corr);
                    // disstence between average and each channel
                    float B_diss=exp(-abs(diss0-aver_corr));
                    float G_diss=exp(-abs(diss1-aver_corr));
                    float R_diss=exp(-abs(diss2-aver_corr));
                    CO_weight[0]=B_diss/(B_diss+G_diss+R_diss);
                    CO_weight[1]=G_diss/(B_diss+G_diss+R_diss);
                    CO_weight[2]=R_diss/(B_diss+G_diss+R_diss);
                    // 计算不同视差下的 sensus 距离
                    //float hi=(float_t)left_census.census_hanming_dist(CensusLeftB.at<double>(i,j),CensusRightB.at<double>(i,current_right));
                    float census_hanming_B=(float_t)left_census.census_hanming_dist(CensusLeftB.at<double>(i,j),CensusRightB.at<double>(i,current_right));
                    float census_hanming_G=(float_t)left_census.census_hanming_dist(CensusLeftG.at<double>(i,j),CensusRightG.at<double>(i,current_right));
                    float census_hanming_R=(float_t)left_census.census_hanming_dist(CensusLeftR.at<double>(i,j),CensusRightR.at<double>(i,current_right));
                    float aver_census=(census_hanming_B+census_hanming_G+census_hanming_R)/3;
                    float BB_diss=exp(-abs(census_hanming_B-aver_census));
                    float GG_diss=exp(-abs(census_hanming_G-aver_census));
                    float RR_diss=exp(-abs(census_hanming_R-aver_census));
                    CT_weight[0]=BB_diss/(BB_diss+GG_diss+RR_diss);
                    CT_weight[1]=GG_diss/(BB_diss+GG_diss+RR_diss);
                    CT_weight[2]=RR_diss/(BB_diss+GG_diss+RR_diss);
                    float diss_corr=CO_weight[0]/diss0+CO_weight[1]/diss1+CO_weight[2]/diss2;
                    // cout<<" diss: "<<diss_corr<<endl;
                    float diss_ct=CT_weight[0]*census_hanming_B+CT_weight[1]*census_hanming_G+CT_weight[2]*census_hanming_R;
                    float diss=0.5*diss_ct+0.5*diss_corr;
                    nums1.push_back(diss_ct);
                    nums2.push_back(diss_corr);
                    //float diss=diss_ct;
                    // printf("diss : %f \n",diss);
                    if(isnan(diss))
                        cost_res[image_y][image_x][m-MIN_DISPARITY]=FLT_MAX/2;
                    else
                        cost_res[image_y][image_x][m-MIN_DISPARITY]=diss;
                }
                else{
                    cost_res[image_y][image_x][m-MIN_DISPARITY]=FLT_MAX/2;
                }
            }
        }
    }
}
