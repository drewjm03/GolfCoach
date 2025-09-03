// SampleConsoleApplication_VS2017.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "eCAMFwSw.h"
#include <strsafe.h>
#include <stdio.h>


void PrintMessage(LPTSTR szFormat, ...);
TCHAR **szInstanceID;


int _tmain(int argc, _TCHAR* argv[])
{
	int Option = 0;
	int SelectedDeviceIndex = 0;
	CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
	UINT8 MajorVersion, MinorVersion1, SceneMode;
	UINT16 MinorVersion2, MinorVersion3;
	UINT32 DevicesCnt = 0;
	TCHAR *DeviceName = NULL;
	char CamName[MAX_PATH];

ENUMERATE:
	DevicesCnt = 0;
	szInstanceID = NULL;
	printf("\n\t0. Exit\n");
	printf("\t1. Enumerate\n");
	if (GetDevicesCount(&DevicesCnt))
	{
		szInstanceID = (TCHAR**)malloc(DevicesCnt); // Getting num of  devices connected. 

		for (int i = 0; i < DevicesCnt; i++)
		{
			// Allocating memory to store the instance ID devices.
			*(szInstanceID + i) = (TCHAR*)malloc(MAX_PATH * sizeof(TCHAR));
		}

		DeviceName = (TCHAR*)malloc(MAX_PATH * sizeof(TCHAR));

		if (!GetDeviceName(DeviceName)) // Getting device friendly name.
		{
			printf("GetDeviceName failed... \n ");
			exit(0);
		}

		if (!GetDevicePaths(szInstanceID))  // Getting all  devices instance IDs
		{
			printf("GetDevicePaths failed... \n ");
			exit(0);
		}


		for (int cnt = 1; cnt < DevicesCnt + 1; cnt++)
		{
			printf("\t%d. ", (cnt + 1));
			wcstombs(CamName, DeviceName, wcslen(DeviceName) + 1);
			printf("%s", CamName);
		}

		// Now defining the char pointer to be a buffer for wcstomb/wcstombs


		printf("\nChoose the device index : ");
		scanf("%d", &Option);

		if (Option == 0)
			exit(0);
		if (Option == 1)
			goto ENUMERATE;
		if (Option<1 || Option > DevicesCnt + 1)
		{
			printf("\nPlease Choose the Valid Option... :(\n");
			goto ENUMERATE;
		}
		else
			SelectedDeviceIndex = Option - 1;


		for (int i = 2; i < SelectedDeviceIndex; i++)
			*szInstanceID++;



	Start:
		DeinitExtensionUnit();
		if (InitExtensionUnit(*szInstanceID))
		{

			printf("\nOptions:\n  0.Exit\n  1.Read Firmware Version\n  2.Get Unique ID\n  3.Reset Device\n  4.Back\n\nEnter your option : ");
			scanf("%d", &Option);

			switch (Option)
			{
			case 0:
				goto EXIT;
			case 1:
			{
				if (ReadFirmwareVersion(&MajorVersion, &MinorVersion1, &MinorVersion2, &MinorVersion3))
					printf("\n\tFirmware Version is : %d.%d.%d.%d\r\n", MajorVersion, MinorVersion1, MinorVersion2, MinorVersion3);

				else
					printf("ReadFirmwareVersion Failed\n");
				goto Start;
			}
			case 2:
			{
				TCHAR UniqID[MAX_PATH];
				if (GetCameraUniqueID(UniqID))
				{
					printf("\n\tCamera UniqueID : ");
					for (int i = 0; UniqID[i]; i++)
					{
						printf("%c", UniqID[i]);
					}
					printf("\n");
				}
				else
					printf("\r\nGetCameraUniqueID Failed\r\n");
			}
			goto Start;
			case 3:
			{
				if (ResetDevice())
				{
					Sleep(5000);
					printf("\r\ResetDevice success\r\n");
				}
				else
				{
					printf("\r\ResetDevice Failed\r\n");
				}
				goto ENUMERATE;
				break;
			}
			case 4:
			{
				goto ENUMERATE;
			}
			default:
			{
				printf("Invalid Option \n");
				goto Start;
			}
			}
		}
		else
			printf("InitExtensionUnit Failed");
	}
	else
	{
		printf("No device found !");
		scanf("%d", &Option);
		goto ENUMERATE;
	}

EXIT:
	DeinitExtensionUnit();
	CoUninitialize();

	return 0;
}

void PrintMessage(LPTSTR szFormat, ...)
{
	try
	{
		static TCHAR szBuffer[2048] = { 0 };
		const size_t NUMCHARS = sizeof(szBuffer) / sizeof(szBuffer[0]);
		const int LASTCHAR = NUMCHARS - 1;

		// Format the input string
		va_list pArgs;
		va_start(pArgs, szFormat);

		// Use a bounded buffer size to prevent buffer overruns.  Limit count to
		// character size minus one to allow for a NULL terminating character.
		HRESULT hr = StringCchVPrintf(szBuffer, NUMCHARS - 1, szFormat, pArgs);
		va_end(pArgs);

		// Ensure that the formatted string is NULL-terminated
		szBuffer[LASTCHAR] = TEXT('\0');

		OutputDebugStringW(szBuffer);
	}
	catch (...)
	{
		OutputDebugString(TEXT("Exception PrintMessage....\r\n"));
	}
}


